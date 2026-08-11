"""Does the seed determine the OUTCOME, not just the action?

DSRL can only help if the noise sequence materially changes how many cubes get
sorted. Run the frozen policy N times on ONE fixed layout: physics and
arrangement are held constant, so the only thing differing between episodes is
the noise the model samples. The spread in cubes sorted is therefore the
seed-attributable variance, which is the headroom DSRL has to exploit.

Do NOT hold a single seed for a whole episode — that starves the policy of the
per-chunk variation it was trained with and measures a failure mode DSRL never
enters, since DSRL picks a fresh seed each chunk.

    python -m dsrl.eval.seed_leverage --episodes 8 --layout 0
"""

import argparse

import numpy as np

from dsrl.config import DsrlConfig
from dsrl.env.stack_env import StackEnv


def run_episode(env, action):
    env.reset()
    total = 0.0
    while True:
        _, r, done, trunc, info = env.step(action)
        total += r
        if done or trunc:
            return total, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--layout", type=int, default=0)
    args = ap.parse_args()

    cfg = DsrlConfig()
    env = StackEnv(cfg, seed=0)

    print(f"\n{'=' * 66}")
    print(f"  SEED LEVERAGE - layout {args.layout} fixed, {args.episodes} episodes")
    print(f"{'=' * 66}")
    print("  frozen policy, fresh noise each chunk, identical layout every episode")
    print("  spread in 'correct' = variance the noise alone produces = DSRL headroom\n")

    corrects, rewards = [], []
    for k in range(args.episodes):
        env.force_layout(args.layout)
        total, info = run_episode(env, None)
        corrects.append(info["correct"])
        rewards.append(total)
        print(f"  ep {k}: correct={info['correct']} wrong={info['wrong']} "
              f"reward={total:+.1f} {info['elapsed']}s", flush=True)

    c = np.array(corrects)
    print(f"\n{'=' * 66}")
    print(f"  cubes sorted     : {c.tolist()}")
    print(f"  min / mean / max : {c.min()} / {c.mean():.2f} / {c.max()}")
    print(f"  spread           : {c.max() - c.min()}    std: {c.std():.2f}")
    print(f"  reward range     : {min(rewards):+.1f} .. {max(rewards):+.1f}")
    print(f"  headroom if DSRL always picked the best noise: "
          f"{c.mean():.2f} -> {c.max()} cubes")
    print()
    spread = c.max() - c.min()
    if spread >= 2:
        print("  VERDICT: seed strongly determines outcome -> DSRL has real headroom")
    elif spread == 1:
        print("  VERDICT: weak effect -> DSRL could win ~1 cube at best; marginal")
    else:
        print("  VERDICT: seed does not change the outcome -> DSRL cannot help")
    print(f"{'=' * 66}\n")


if __name__ == "__main__":
    main()
