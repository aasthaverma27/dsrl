"""SAC over the diffusion-seed action space.

The action is the noise the frozen policy denoises from, so the actor's output is
a point in seed space rather than a joint command. Standard SAC otherwise: twin
critics, tanh-squashed Gaussian actor, automatic entropy tuning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

def mlp(sizes, out_dim, layer_norm=False, activation="tanh"):
    """Reference architecture: Tanh activation with LayerNorm after each linear, on
    both actor and critic (train_dsrl.py passes activation_fn=nn.Tanh and
    post_linear_modules=[LayerNorm] via net_arch=dict(pi=..., qf=...))."""
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    layers, last = [], sizes[0]
    for h in sizes[1:]:
        layers.append(nn.Linear(last, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, layer_norm=True, activation="tanh"):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], 2 * act_dim,
                       layer_norm=layer_norm, activation=activation)
        self.act_dim = act_dim
        # Start the composite policy AT the base policy. DSRL's guarantee that the
        # optimum is no worse than the frozen policy holds because feeding i.i.d.
        # N(0,1) reproduces it exactly -- but only if that point is reachable and,
        # in practice, only helps if we start there. Zeroing the output layer gives
        # mu=0 and log_std=0, so the initial policy is a state-independent draw.
        # log_std=0 also matches the reference's log_std_init=0.0. A random init
        # instead begins at an arbitrary state-dependent point and has to climb back
        # to baseline before it can improve on it.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        return mu, log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs):
        """Tanh-squashed Gaussian over the box, as in the reference.

        Their noise space is Box(-action_magnitude, +action_magnitude) with a standard
        SAC actor (env_utils.py:199-200), and _build_seed scales this output by
        noise_scale=1.5. Removing the squash and projecting onto a norm shell was our
        own construction, not theirs.
        """
        mu, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        action = torch.tanh(x)
        # tanh change-of-variables correction
        logp = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, logp.sum(-1, keepdim=True)


class Critic(nn.Module):
    """Ensemble of num_critics Q heads; the reference uses 2 with a `min` backup."""

    def __init__(self, obs_dim, act_dim, hidden, num_critics=2, activation="tanh"):
        super().__init__()
        self.qs = nn.ModuleList(
            [mlp([obs_dim + act_dim, *hidden], 1, layer_norm=True, activation=activation)
             for _ in range(num_critics)]
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return torch.cat([q(x) for q in self.qs], dim=-1)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros((capacity, 1), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.capacity, self.idx, self.size = capacity, 0, 0

    def add(self, o, a, r, no, d):
        i = self.idx
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, no, d
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, device):
        j = np.random.randint(0, self.size, size=batch)
        out = (self.obs[j], self.act[j], self.rew[j], self.next_obs[j], self.done[j])
        return [torch.as_tensor(x, device=device) for x in out]


class SAC:
    def __init__(self, obs_dim, act_dim, cfg, device="cuda:0"):
        self.device = torch.device(device)
        hidden = tuple(cfg.hidden)
        n_q = cfg.num_critics
        self.reduction = cfg.critic_reduction
        self.subset_size = getattr(cfg, "critic_subset_size", 2)
        act_fn = getattr(cfg, "activation", "tanh")
        ln_actor = getattr(cfg, "layer_norm_actor", True)
        self.actor = Actor(obs_dim, act_dim, hidden, ln_actor, act_fn).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden, n_q, act_fn).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden, n_q, act_fn).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Sign matters and is easy to get backwards: alpha equilibrates where
        # entropy == target_entropy, so a NEGATIVE target asks for a concentrated
        # policy. Small control tasks want that (-1/dim); we do not, because the
        # base policy we must not fall below is itself stochastic.
        self.target_entropy = float(act_dim) * cfg.target_entropy_per_dim
        self.alpha_max = getattr(cfg, "alpha_max", 1.0)
        self.q_bound = getattr(cfg, "q_bound", 60.0)
        self.log_alpha = torch.full(
            (1,), float(np.log(cfg.init_alpha)), device=self.device, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

        with torch.no_grad():
            _, lp = self.actor.sample(torch.zeros(256, obs_dim, device=self.device))
        h_base = -float(lp.mean())
        print(
            f"[sac] entropy at base-policy init {h_base:+.1f} | target "
            f"{self.target_entropy:+.1f} | alpha {float(self.alpha):.3f} | "
            f"entropy term {float(self.alpha) * abs(h_base):.1f} vs reward scale ~10"
        )
        if self.target_entropy > h_base:
            print("[sac] WARNING: target entropy exceeds what this policy can reach")

        self.gamma, self.tau = cfg.gamma, cfg.tau
        self.buffer = ReplayBuffer(obs_dim, act_dim, cfg.replay_capacity)

    def _reduce(self, qs: torch.Tensor) -> torch.Tensor:
        """Pessimistic reduction over the critic ensemble.

        "mean" gives no pessimism at all, and plain min over all heads is barely
        better because every head trains on identical targets and they converge
        together. REDQ's fix is min over a random subset each update: the sampling
        keeps the heads disagreeing, which is what actually bounds overestimation.
        """
        if self.reduction == "mean":
            return qs.mean(-1, keepdim=True)
        if self.reduction == "min":
            return qs.min(-1, keepdim=True).values
        # "subset" (REDQ): min over subset_size randomly chosen heads
        idx = torch.randperm(qs.shape[-1], device=qs.device)[: self.subset_size]
        return qs[..., idx].min(-1, keepdim=True).values

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if deterministic:
            mu, _ = self.actor(o)
            return torch.tanh(mu).squeeze(0).cpu().numpy()
        a, _ = self.actor.sample(o)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch_size):
        obs, act, rew, next_obs, done = self.buffer.sample(batch_size, self.device)

        with torch.no_grad():
            next_act, _ = self.actor.sample(next_obs)
            qs_t = self.critic_target(next_obs, next_act)
            q_next = self._reduce(qs_t)
            # entropy stays in the actor loss only; q_bound is the provable return bound
            target = rew + (1 - done) * self.gamma * q_next
            target = target.clamp(-self.q_bound, self.q_bound)

        qs = self.critic(obs, act)
        critic_loss = F.mse_loss(qs, target.expand_as(qs))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        new_act, logp = self.actor.sample(obs)
        qs_pi = self.critic(obs, new_act)
        actor_loss = (self.alpha * logp - self._reduce(qs_pi)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha.exp() * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(max=float(np.log(self.alpha_max)))

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha": float(self.alpha),
            "q": float(qs.mean()),
        }
