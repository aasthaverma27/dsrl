"""GR00T baseline on the in-process sort env (run via scripts/baseline.sh).

Modes:
  native     seed=None            -> GR00T samples its own torch.randn (stock policy)
  stochastic random actor action  -> routed through env.step's exact mapping
                                     (clip +-1, x noise_scale, residual/replace per
                                     cfg.groot.steer_residual) -- the exploration floor
  determin   the no-op action     -> steer=None: the episode's base seed untouched
                                     (was base+zeros before F18/F19 -- degenerate)
"""
import os, sys, time, json, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["native", "stochastic", "deterministic"], default="native")
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--layout", type=int, default=0)
ap.add_argument("--max-chunks", type=int, default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

from dsrl.config import DsrlConfig
from dsrl.env.sort_env import CubeSortEnv

cfg = DsrlConfig()
env = CubeSortEnv(cfg, seed=args.seed)
rng = np.random.default_rng(args.seed + 1000)
max_chunks = args.max_chunks or cfg.env.max_chunks
recs = []

for ep in range(args.episodes):
    t0 = time.time()
    env.reset(config_idx=args.layout)
    for c in range(max_chunks):
        if args.mode == "native":
            seed = None                       # GR00T's own randn
        elif args.mode == "stochastic":
            action = rng.standard_normal(env.policy.action_dim).astype(np.float32)
            steer = np.clip(action, -1.0, 1.0) * cfg.groot.noise_scale
            seed = env.policy.build_seed(env.base_noise, steer,
                                         residual=cfg.groot.steer_residual)
        else:
            seed = env.policy.build_seed(env.base_noise, None)

        obs_in = env.policy.observe(env.joint_pos, env.render())
        chunk = env.policy.chunk(obs_in, seed)
        for k in range(cfg.env.execute_steps):
            env._run(env._slice_chunk(chunk, k), env.sim_per_ctrl)
        env.chunks += 1

        score = env.placements()
        if score["correct"] >= cfg.env.n_cubes or env.pelvis_height < cfg.reward.fall_height:
            break

    score = env.placements()
    rec = {"ep": ep, "mode": args.mode, "layout": args.layout,
           "correct": int(score["correct"]), "wrong": int(score["wrong"]),
           "fell": bool(env.pelvis_height < cfg.reward.fall_height),
           "chunks": env.chunks, "secs": round(time.time() - t0, 1)}
    recs.append(rec)
    print("  " + json.dumps(rec), flush=True)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

n = len(recs)
tot = sum(r["correct"] for r in recs)
full = sum(1 for r in recs if r["correct"] >= 3)
print("\nSUMMARY mode=%s n=%d  per-cube=%.1f%%  full(3/3)=%.1f%%  wrong=%d  fell=%d  mean_secs=%.1f"
      % (args.mode, n, 100*tot/(3*n), 100*full/n,
         sum(r["wrong"] for r in recs), sum(r["fell"] for r in recs),
         sum(r["secs"] for r in recs)/n))
