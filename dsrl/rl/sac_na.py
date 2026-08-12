"""DSRL-NA: noise aliasing. Paper Algorithm 1, adapted to a 1.3B action head.

Two critics instead of one:

  Q^A(s, a)  over ENVIRONMENT actions, trained by TD. The reward is a function of the
             executed motion, so this is the space where the value function is smooth.
  Q^W(s, w)  over NOISE, trained by supervised REGRESSION toward Q^A(s, pi_dp(s,w)).
             No bootstrapping, no sparse reward -- a dense target for every sample.

The actor optimises Q^W. This is the mechanism that addresses our measured failure:
the noise critic previously learned by TD from a reward firing ~12 times per episode
and reached cal_corr 0.15; here it learns by regression against an already-fitted
action-space critic.

TWO DOCUMENTED DEVIATIONS from Algorithm 1, both forced by the model's size:

1. The Q^A target action. The paper uses a' ~ pi_dp(s', pi^W(s')), which needs one
   denoise per batch element -- at batch 256 and utd 10 that is 2,560 denoises per
   env step through a 1.09B DiT (52 ms each even batched), which does not fit in a
   400 ms step. We use the action actually executed at s', which is already in the
   buffer. This makes the target SARSA-style rather than an off-policy max: biased
   toward the behaviour policy, but free.

2. The (w, a) pairs for distillation come from collection, not from the training loop.
   The policy server denoises K extra seeds off the same backbone pass, measured at
   +7 ms for 16 candidates (the DiT is bandwidth-bound, so batching is ~free). So the
   pairs are sampled once per real state rather than freshly each update.
"""

import numpy as np
import torch
import torch.nn.functional as F

from dsrl.rl.sac import SAC, Critic


class NaReplayBuffer:
    """Adds, per transition, the executed action chunk and K (noise, action) pairs.

    Pairs are float16: at K=8 they are 8*(56+140) values per step, which at 200k
    capacity is ~630 MB. Precision does not matter here -- they are regression
    inputs, and we measured the denoise itself is only bf16-accurate anyway.
    """

    def __init__(self, obs_dim, act_dim, env_act_dim, capacity, k_pairs):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros((capacity, 1), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.env_act = np.zeros((capacity, env_act_dim), np.float32)
        self.pair_w = np.zeros((capacity, k_pairs, act_dim), np.float16)
        self.pair_a = np.zeros((capacity, k_pairs, env_act_dim), np.float16)
        self.has_pairs = np.zeros(capacity, bool)
        self.capacity, self.idx, self.size = capacity, 0, 0
        self.k_pairs = k_pairs

    def add(self, o, a, r, no, d, env_act=None, pair_w=None, pair_a=None):
        i = self.idx
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, no, d
        if env_act is not None:
            self.env_act[i] = env_act
        # Pairs can be missing: the feedback channel is best-effort and must never
        # stall the control loop, so a dropped record just means this state is not
        # used for distillation.
        ok = pair_w is not None and pair_a is not None
        if ok:
            n = min(len(pair_w), self.k_pairs)
            self.pair_w[i, :n] = pair_w[:n]
            self.pair_a[i, :n] = pair_a[:n]
        self.has_pairs[i] = ok
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, device):
        j = np.random.randint(0, self.size, size=batch)
        # next action for the SARSA-style Q^A target: the action executed at s'.
        # Consecutive indices are consecutive transitions except at the write head
        # and at episode boundaries, where `done` zeroes the bootstrap anyway.
        jn = (j + 1) % max(self.size, 1)
        out = (self.obs[j], self.act[j], self.rew[j], self.next_obs[j], self.done[j],
               self.env_act[j], self.env_act[jn])
        return [torch.as_tensor(x, device=device) for x in out]

    def sample_pairs(self, batch, device):
        """States that carry pairs, with their K (w, a) sets, for the distillation."""
        have = np.flatnonzero(self.has_pairs[: self.size])
        if len(have) == 0:
            return None
        j = have[np.random.randint(0, len(have), size=batch)]
        return [torch.as_tensor(x, device=device) for x in
                (self.obs[j], self.pair_w[j].astype(np.float32),
                 self.pair_a[j].astype(np.float32))]


class SacNA(SAC):
    def __init__(self, obs_dim, act_dim, env_act_dim, cfg, device="cuda:0"):
        super().__init__(obs_dim, act_dim, cfg, device=device)
        hidden = tuple(cfg.hidden)
        act_fn = getattr(cfg, "activation", "tanh")
        self.env_act_dim = env_act_dim
        self.k_pairs = cfg.na_pairs
        self.noise_critic_grad_steps = cfg.noise_critic_grad_steps
        self.distill_batch = cfg.na_distill_batch

        # self.critic / self.critic_target from SAC become Q^A, over ENV actions.
        self.critic = Critic(obs_dim, env_act_dim, hidden, cfg.num_critics, act_fn).to(self.device)
        self.critic_target = Critic(obs_dim, env_act_dim, hidden, cfg.num_critics, act_fn).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Q^W, over noise. Same head count so the distillation can pair them
        # head-by-head, as their update_noise_critic does.
        self.critic_w = Critic(obs_dim, act_dim, hidden, cfg.num_critics, act_fn).to(self.device)
        self.critic_w_opt = torch.optim.Adam(self.critic_w.parameters(), lr=cfg.lr)

        self.buffer = NaReplayBuffer(obs_dim, act_dim, env_act_dim,
                                     cfg.replay_capacity, self.k_pairs)
        print(f"[sac-na] Q^A over env actions ({env_act_dim}d), Q^W over noise ({act_dim}d), "
              f"{cfg.num_critics} heads each, {self.k_pairs} pairs/step, "
              f"{self.noise_critic_grad_steps} distill steps/update", flush=True)

    def update(self, batch_size):
        obs, act, rew, next_obs, done, env_act, next_env_act = self.buffer.sample(batch_size, self.device)

        # --- 1. Q^A: TD in action space -------------------------------------------
        with torch.no_grad():
            _, next_logp = self.actor.sample(next_obs)
            q_next = self._reduce(self.critic_target(next_obs, next_env_act))
            target = rew + (1 - done) * self.gamma * (q_next - self.alpha * next_logp)
        qs_a = self.critic(obs, env_act)
        loss_a = F.mse_loss(qs_a, target.expand_as(qs_a))
        self.critic_opt.zero_grad(set_to_none=True)
        loss_a.backward()
        self.critic_opt.step()

        # --- 2. actor: maximise Q^W (with the entropy term, as in their dsrl.py) ---
        new_act, logp = self.actor.sample(obs)
        actor_loss = (self.alpha * logp - self._reduce(self.critic_w(obs, new_act))).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha.exp() * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return {"critic_loss": float(loss_a), "actor_loss": float(actor_loss),
                "alpha": float(self.alpha), "q": float(qs_a.mean())}

    def distill(self, batch_size=None):
        """Q^W(s,w) -> Q^A(s, pi_dp(s,w)) over the stored pairs. Supervised, no TD.

        Their update_noise_critic pairs the two critics head-by-head rather than
        reducing first, so head i of Q^W learns head i of Q^A.
        """
        batch_size = batch_size or self.distill_batch
        losses = []
        for _ in range(self.noise_critic_grad_steps):
            got = self.buffer.sample_pairs(batch_size, self.device)
            if got is None:
                return {}
            o, w, a = got
            B, K = w.shape[0], w.shape[1]
            o_rep = o.unsqueeze(1).expand(B, K, o.shape[-1]).reshape(B * K, -1)
            with torch.no_grad():
                q_a = self.critic(o_rep, a.reshape(B * K, -1))
            q_w = self.critic_w(o_rep, w.reshape(B * K, -1))
            loss = 0.5 * F.mse_loss(q_w, q_a)
            self.critic_w_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_w_opt.step()
            losses.append(float(loss))
        return {"distill_loss": float(np.mean(losses))}
