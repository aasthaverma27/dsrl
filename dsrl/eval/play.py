"""Run a trained checkpoint (or the frozen policy) against the live stack.

    python -m dsrl.eval.play --ckpt runs/<run>/ckpt_final.pt --episodes 6
    python -m dsrl.eval.play --baseline --episodes 6

--baseline publishes no seed, so their client falls back to sampling its own noise:
that is the frozen checkpoint, and the number DSRL has to beat.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dsrl.config import DsrlConfig
from dsrl.env.stack_env import StackEnv
from dsrl.rl.sac import SAC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--baseline", action="store_true", help="frozen policy, no steering")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--layout", type=int, default=None,
                    help="pin every episode to one arrangement; must match training for a fair A/B")
    # Sampling is the correct default here, unlike most RL eval. The mean action
    # drops the noise, and a seed with no noise in it is one fixed vector per state
    # -- measurably far below baseline. At init mu is exactly 0, which the shell
    # projection turns into an all-zeros seed, the worst possible input.
    ap.add_argument("--deterministic", action="store_true",
                    help="use the mean action (diagnostic only; expect it to underperform)")
    args = ap.parse_args()

    cfg = DsrlConfig()
    env = StackEnv(cfg, seed=0)
    agent = None
    if not args.baseline:
        if args.ckpt is None:
            ap.error("give --ckpt or --baseline")
        agent = SAC(env.obs_dim, env.action_dim, cfg.sac)
        ckpt = torch.load(args.ckpt, map_location=agent.device)
        agent.actor.load_state_dict(ckpt["actor"])
        print(f"[INFO] loaded actor from {args.ckpt} (trained to step {ckpt.get('step')})")

    mode = "GR00T ONLY (no seed steering)" if agent is None else "GR00T + DSRL (seed steered)"
    bar = "=" * 62
    print(f"\n{bar}\n  MODE: {mode}\n{bar}", flush=True)
    if agent is None:
        print("  the model samples its own noise every chunk -> frozen baseline\n", flush=True)
    else:
        print(f"  SAC actor picks the {env.action_dim}-dim seed every chunk\n", flush=True)

    rows = []
    for ep in range(args.episodes):
        if args.layout is not None:
            env.force_layout(args.layout)
        obs = env.reset()
        total, step = 0.0, 0
        print(f"  --- episode {ep} [{mode}] ---", flush=True)
        while True:
            if agent is None:
                action = None  # publish nothing: the model samples its own noise
            else:
                action = agent.act(obs, deterministic=args.deterministic)
            obs, r, done, trunc, info = env.step(action)
            total += r
            step += 1
            if step % 5 == 0 or r != 0:
                src = "randn" if agent is None else "DSRL"
                print(f"    [{src}] step {step:3d} r={r:+6.2f} total={total:+7.2f} "
                      f"correct={info['correct']} wrong={info['wrong']}"
                      + (f" {list(info['events'].keys())}" if info["events"] else ""), flush=True)
            if done or trunc:
                break
        rows.append((info["correct"], info["wrong"], total, info["fell"], info["elapsed"]))
        why = "SORTED ALL 3" if info["correct"] == 3 else ("FELL" if info["fell"] else "time up")
        print(f"  ep{ep} END [{mode}]: correct={info['correct']} wrong={info['wrong']} "
              f"reward={total:+.1f} {info['elapsed']}s -> {why}\n", flush=True)

    c = [r[0] for r in rows]
    label = "BASELINE (frozen)" if args.baseline else f"CKPT {args.ckpt.name}"
    print(f"\n=== {label} over {len(rows)} episodes ===")
    print(f"  mean correct   : {np.mean(c):.2f} / 3")
    print(f"  full sorts     : {sum(x == 3 for x in c)}/{len(rows)}")
    print(f"  >=1 placed     : {sum(x > 0 for x in c)}/{len(rows)}")
    print(f"  mean reward    : {np.mean([r[2] for r in rows]):+.1f}")
    print(f"  falls          : {sum(r[3] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
