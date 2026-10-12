"""Independent n-episode evaluation of a saved actor checkpoint.

Replicates train.py's evaluate() protocol exactly (stochastic actor sampling,
layout 0) but with a larger n and a fresh env/seed, so a hot in-training eval
number can be confirmed or refuted honestly. The reported protocol is
--max-chunks 120 --episodes 30, at seeds 123 and 456. Run via scripts/eval.sh.
"""
import os, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--episodes", type=int, default=30)
ap.add_argument("--layout", type=int, default=0)
ap.add_argument("--max-chunks", type=int, default=90)
ap.add_argument("--seed", type=int, default=123)          # fresh, unused by training
ap.add_argument("--config", default=None, help="run config.json to replicate overrides")
ap.add_argument("--out", default=None)
# Critic-guided seed selection (round 4): sample K candidate actions from the
# actor each chunk and execute the argmax-Q one. Requires a FULL checkpoint
# (actor+critic — the end-of-run or ckpt_best saves). K=1 = plain sampling.
ap.add_argument("--bestof", type=int, default=1)
# candidate scoring: 'mean' amplifies critic optimism under argmax; 'min' is
# pessimistic (a candidate must look good to EVERY head to win)
ap.add_argument("--score", choices=["mean", "min"], default="mean")
args = ap.parse_args()

from dsrl.config import DsrlConfig
from dsrl.rl.train import apply_overrides
from dsrl.env.sort_env import CubeSortEnv
from dsrl.rl.sac import SAC

cfg = DsrlConfig()
if args.config:
    # replicate the run's groot/env settings from its config.json
    saved = json.loads(open(args.config).read())
    sets = []
    for grp in ("groot", "env", "sac"):
        for k, v in saved.get(grp, {}).items():
            cur = getattr(getattr(cfg, grp), k, None)
            if cur is not None and not isinstance(cur, (tuple, list)) and cur != v:
                sets.append(f"{grp}.{k}={v}")
    if sets:
        cfg = apply_overrides(cfg, tuple(sets))
        print("[eval] replicated overrides:", sets)

# the CLI episode budget wins over the run config's, or the env truncates early
cfg = apply_overrides(cfg, (f"env.max_chunks={args.max_chunks}",))

env = CubeSortEnv(cfg, seed=args.seed)
if cfg.sac.use_pixels:
    from dsrl.rl.sac_pixel import PixelSAC

    agent = PixelSAC(env.obs_dim, env.action_dim, cfg.sac)
else:
    agent = SAC(env.obs_dim, env.action_dim, cfg.sac)
ckpt = torch.load(args.ckpt, map_location=agent.device)
agent.actor.load_state_dict(ckpt["actor"])
if hasattr(agent, "encoder"):
    agent.encoder.load_state_dict(ckpt["encoder"])
print(f"[eval] loaded actor from {args.ckpt} (step {ckpt.get('step')})")


def policy_act(obs):
    if hasattr(agent, "encoder"):
        return agent.act(obs, pixels=env.last_pixels)
    return agent.act(obs)
if args.bestof > 1:
    if "critic" not in ckpt:
        raise SystemExit("--bestof needs a full checkpoint (actor+critic)")
    agent.critic.load_state_dict(ckpt["critic"])
    print(f"[eval] critic loaded; best-of-{args.bestof} seed selection active")


def select_action(obs):
    if args.bestof <= 1:
        return policy_act(obs)
    with torch.no_grad():
        o = torch.as_tensor(obs, device=agent.device, dtype=torch.float32)
        o = o.unsqueeze(0).expand(args.bestof, -1)
        acts, _ = agent.actor.sample(o)
        qs = agent.critic(o, acts)
        q = qs.min(-1).values if args.score == "min" else qs.mean(-1)
        return acts[q.argmax()].cpu().numpy()

recs = []
for ep in range(args.episodes):
    obs = env.reset(config_idx=args.layout)
    info = {}
    for _ in range(args.max_chunks):
        obs, r, done, trunc, info = env.step(select_action(obs))
        if done or trunc:
            break
    rec = {"ep": ep, "correct": int(info["correct"]), "wrong": int(info["wrong"]),
           "fell": bool(info["fell"]), "chunks": env.chunks}
    recs.append(rec)
    print("  " + json.dumps(rec), flush=True)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

n = len(recs)
tot = sum(r["correct"] for r in recs)
full = sum(1 for r in recs if r["correct"] >= 3)
print("CONFIRM ckpt=%s n=%d per-cube=%.1f%% full=%.1f%% wrong=%d fell=%d"
      % (os.path.basename(args.ckpt), n, 100 * tot / (3 * n), 100 * full / n,
         sum(r["wrong"] for r in recs), sum(r["fell"] for r in recs)), flush=True)
