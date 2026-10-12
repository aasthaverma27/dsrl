"""PixelSAC: SAC over the diffusion-seed space with the ego view as input.

Matches the reference DSRL implementation's design point — their pi0 agent is
`pixel_sac`: raw camera pixels through a learned CNN, proprio concatenated
(github nakamotoo/dsrl_pi0). At our sample scale (tens of thousands of
transitions, not millions) we use the DrQ-v2 recipe instead of their
ResNet-34: a small conv encoder trained by the critic loss only, the actor
consuming detached features, and pad-4 random-crop augmentation. Everything
seed-space-specific (zero-init actor head, REDQ subset-min, q_bound) is
inherited from SAC.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dsrl.rl.sac import SAC, ReplayBuffer


class Encoder(nn.Module):
    """DrQ conv stack for 84x84 uint8 RGB -> pixel_latent features."""

    def __init__(self, latent_dim: int, size: int = 84):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3, size, size)).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Linear(n, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh())

    def forward(self, px):
        # px: (B, H, W, 3) uint8 -> (B, 3, H, W) float in [-0.5, 0.5]
        x = px.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.head(self.conv(x).flatten(1))


def random_crop(px: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ pad-and-crop on a (B, H, W, C) uint8 batch, applied per-sample."""
    b, h, w, c = px.shape
    x = px.permute(0, 3, 1, 2).float()
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    out = torch.empty_like(px, dtype=torch.float32).permute(0, 3, 1, 2)
    ii = torch.randint(0, 2 * pad + 1, (b,), device=px.device)
    jj = torch.randint(0, 2 * pad + 1, (b,), device=px.device)
    for k in range(b):
        out[k] = x[k, :, ii[k]:ii[k] + h, jj[k]:jj[k] + w]
    return out.permute(0, 2, 3, 1).to(torch.uint8)


class PixelReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, capacity, size):
        super().__init__(obs_dim, act_dim, capacity)
        self.px = np.zeros((capacity, size, size, 3), np.uint8)
        self.next_px = np.zeros((capacity, size, size, 3), np.uint8)

    def add(self, o, a, r, no, d, px=None, next_px=None):
        i = self.idx
        self.px[i], self.next_px[i] = px, next_px
        super().add(o, a, r, no, d)

    def sample(self, batch, device):
        j = np.random.randint(0, self.size, size=batch)
        out = (self.obs[j], self.act[j], self.rew[j], self.next_obs[j], self.done[j])
        ts = [torch.as_tensor(x, device=device) for x in out]
        ts.append(torch.as_tensor(self.px[j], device=device))
        ts.append(torch.as_tensor(self.next_px[j], device=device))
        return ts


class PixelSAC(SAC):
    def __init__(self, obs_dim, act_dim, cfg, device="cuda:0"):
        # actor/critic MLPs see [proprio, pixel_latent]
        super().__init__(obs_dim + cfg.pixel_latent, act_dim, cfg, device)
        self.raw_obs_dim = obs_dim
        self.encoder = Encoder(cfg.pixel_latent, cfg.pixel_size).to(self.device)
        # the encoder learns from the critic loss only (DrQ-v2)
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.encoder.parameters()), lr=cfg.lr
        )
        self.buffer = PixelReplayBuffer(
            obs_dim, act_dim, cfg.pixel_replay_capacity, cfg.pixel_size
        )
        self._last_px = None

    @torch.no_grad()
    def act(self, obs, deterministic=False, pixels=None):
        if pixels is not None:
            self._last_px = pixels
        px = torch.as_tensor(self._last_px, device=self.device).unsqueeze(0)
        feat = self.encoder(px)
        o = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        o = torch.cat([o, feat], dim=-1)
        if deterministic:
            mu, _ = self.actor(o)
            return torch.tanh(mu).squeeze(0).cpu().numpy()
        a, _ = self.actor.sample(o)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch_size):
        obs, act, rew, next_obs, done, px, next_px = self.buffer.sample(batch_size, self.device)
        px, next_px = random_crop(px), random_crop(next_px)

        with torch.no_grad():
            next_in = torch.cat([next_obs, self.encoder(next_px)], dim=-1)
            next_act, _ = self.actor.sample(next_in)
            q_next = self._reduce(self.critic_target(next_in, next_act))
            target = rew + (1 - done) * self.gamma * q_next
            target = target.clamp(-self.q_bound, self.q_bound)

        feat = self.encoder(px)
        obs_in = torch.cat([obs, feat], dim=-1)
        qs = self.critic(obs_in, act)
        critic_loss = F.mse_loss(qs, target.expand_as(qs))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        obs_pi = torch.cat([obs, feat.detach()], dim=-1)
        new_act, logp = self.actor.sample(obs_pi)
        actor_loss = (self.alpha * logp - self._reduce(self.critic(obs_pi, new_act))).mean()
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
