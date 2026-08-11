"""Watch the frozen policy (or a steered one) drive the robot, viewer up.

    python -m dsrl.eval.watch --episodes 3 --chunks 66 --mode unsteered

Attaches a viewer to the env's own loop rather than reimplementing it, so what you
see is exactly what training runs.
"""

import argparse

import mujoco.viewer
import numpy as np

from dsrl.config import DsrlConfig
from dsrl.env.sort_env import CubeSortEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--chunks", type=int, default=66)
    ap.add_argument("--mode", choices=["unsteered", "random"], default="unsteered")
    ap.add_argument("--fast", action="store_true", help="run flat out instead of realtime")
    args = ap.parse_args()

    cfg = DsrlConfig()
    env = CubeSortEnv(cfg, seed=0)
    hand_bid = env.scene.model.body("right_wrist_yaw_link").id
    rng = np.random.default_rng(0)

    with mujoco.viewer.launch_passive(env.scene.model, env.scene.data) as viewer:
        env.viewer = viewer
        env.realtime = not args.fast

        for ep in range(args.episodes):
            env.reset(ep % 6)
            print(f"\n--- episode {ep} (cube config {ep % 6}) ---", flush=True)
            for k in range(args.chunks):
                if not viewer.is_running():
                    return
                action = (
                    np.zeros(env.action_dim, np.float32)
                    if args.mode == "unsteered"
                    else rng.uniform(-1, 1, env.action_dim).astype(np.float32)
                )
                _, r, done, trunc, info = env.step(action)
                d = min(
                    np.linalg.norm(env.scene.cube_pos(i) - env.scene.data.xpos[hand_bid])
                    for i in range(3)
                )
                if k % 5 == 0 or r != 0:
                    print(
                        f"  chunk {k:3d}  reward={r:+.1f}  correct={info['correct']} "
                        f"wrong={info['wrong']}  nearest cube={d:.3f}m  "
                        f"pelvis={env.scene.pelvis_height:.3f}",
                        flush=True,
                    )
                if done or trunc:
                    break
            print(f"  END correct={info['correct']} wrong={info['wrong']} fell={info['fell']}",
                  flush=True)


if __name__ == "__main__":
    main()
