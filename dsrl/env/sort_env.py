"""Gym environment where the action is the frozen policy's diffusion seed.

Everything robot-side is gear_sonic's and gr00t_wbc's own code:
  scene, cubes, cameras, reset  -> gear_sonic DefaultEnv
  balance + upper-body goals    -> gr00t_wbc G1DecoupledWholeBodyPolicy (WbcAdapter)
  action chunks                 -> the frozen GR00T checkpoint (SeededGrootPolicy)

DSRL adds only the seed, the reward, and episode bookkeeping.
"""

from pathlib import Path
import time

import numpy as np

from dsrl.groot.policy import SeededGrootPolicy
from dsrl.sim.wbc_adapter import WbcAdapter, N_JOINTS

HEIGHT_CMD = 0.74
WBC_DECIMATION = 4
COLORS = ("red", "green", "blue")

INIT_POSE = np.load(Path(__file__).resolve().parent.parent / "sim/init_pose.npy")


def pose_as_chunk(pose: np.ndarray) -> dict:
    """A static 43-dim pose in the (B, 1, D) chunk shape their goal builder reads."""
    return {
        "left_arm": pose[15:22][None, None, :],
        "left_hand": pose[22:29][None, None, :],
        "right_arm": pose[29:36][None, None, :],
        "right_hand": pose[36:43][None, None, :],
        "base_height_command": np.array([[[HEIGHT_CMD]]]),
        "navigate_command": np.zeros((1, 1, 3)),
    }


def build_their_env(offscreen: bool = True):
    """Instantiate gear_sonic's DefaultEnv exactly as run_sim_loop.py does."""
    from gear_sonic.utils.mujoco_sim.base_sim import DefaultEnv
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    cfg = SimLoopConfig()
    cfg.enable_offscreen = offscreen
    wbc_config = cfg.load_wbc_yaml()
    wbc_config["ENV_NAME"] = cfg.env_name
    # enable_image_publish only gates their default camera_configs here; the ZMQ
    # publisher is a separate explicit call we never make.
    return DefaultEnv(
        config=wbc_config,
        env_name=cfg.env_name,
        onscreen=False,
        offscreen=offscreen,
        enable_image_publish=True,
    )


class CubeSortEnv:
    def __init__(self, cfg, policy: SeededGrootPolicy | None = None, seed: int = 0):
        self.cfg = cfg
        self.sim = build_their_env(offscreen=True)
        self.model, self.data = self.sim.mj_model, self.sim.mj_data
        self.wbc = WbcAdapter()
        self.policy = policy if policy is not None else SeededGrootPolicy(
            steerable=cfg.groot.steerable_groups
        )
        self.rng = np.random.default_rng(seed)

        self.cube_bid = [self.model.body(f"cube_{c}").id for c in COLORS]
        self.basket_bid = [self.model.body(f"basket_{c}").id for c in COLORS]
        self.hand_bid = self.model.body("right_wrist_yaw_link").id

        gid = self.model.geom("basket_red_wpx").id
        self.basket_inner = float(self.model.geom_pos[gid][0]) - float(self.model.geom_size[gid][0])
        floor = self.model.geom("basket_red_floor").id
        self.cube_half = float(self.model.geom("cube_red_geom").size[0])
        self.basket_floor_z = float(
            self.model.geom_pos[floor][2] + self.model.geom_size[floor][2]
        )

        self.sim_per_ctrl = int(round(1.0 / cfg.env.control_hz / self.model.opt.timestep))
        self.action_dim = self.policy.action_dim
        self.episode = 0
        self.viewer = None
        self.realtime = False

    # ---------- task state ----------

    def cube_pos(self, i):
        return self.data.xpos[self.cube_bid[i]].copy()

    def basket_pos(self, i):
        return self.data.xpos[self.basket_bid[i]].copy()

    @property
    def joint_pos(self):
        return self.data.qpos[7 : 7 + N_JOINTS].copy()

    @property
    def pelvis_height(self):
        return float(self.data.qpos[2])

    def progress_distance(self) -> float:
        """Dense task-progress measure: how far the scene is from being sorted.

        Two terms, both in metres: how far the hand is from the nearest cube that
        still needs moving, and how far each unplaced cube is from its own basket.
        Rewarding the reduction gives a signal every step instead of one bit per
        episode, which placement-only reward cannot provide.
        """
        hand = self.data.xpos[self.hand_bid]
        placed = self._placed_mask()
        remaining = [i for i in range(3) if not placed[i]]
        if not remaining:
            return 0.0
        reach = min(float(np.linalg.norm(self.cube_pos(i) - hand)) for i in remaining)
        transport = sum(
            float(np.linalg.norm(self.cube_pos(i) - self.basket_pos(i))) for i in remaining
        )
        return reach + transport

    def _placed_mask(self):
        return [
            any(self._in_basket(i, j) for j in range(3)) for i in range(3)
        ]

    def _in_basket(self, i, j) -> bool:
        c, b = self.cube_pos(i), self.basket_pos(j)
        rest_z = b[2] + self.basket_floor_z + self.cube_half
        return bool(
            np.abs(c[:2] - b[:2]).max() < self.basket_inner
            and abs(c[2] - rest_z) < 2 * self.cube_half
        )

    def placements(self):
        correct = wrong = 0
        for i in range(3):
            for j in range(3):
                c, b = self.cube_pos(i), self.basket_pos(j)
                rest_z = b[2] + self.basket_floor_z + self.cube_half
                if (np.abs(c[:2] - b[:2]).max() < self.basket_inner
                        and abs(c[2] - rest_z) < 2 * self.cube_half):
                    correct += i == j
                    wrong += i != j
                    break
        return {"correct": correct, "wrong": wrong}

    def render(self):
        return self.sim.update_render_caches()["ego_view_image"]

    def observation(self) -> np.ndarray:
        hand = self.data.xpos[self.hand_bid]
        rel = [self.cube_pos(i) - self.basket_pos(i) for i in range(3)]
        to_cubes = [self.cube_pos(i) - hand for i in range(3)]
        return np.concatenate(
            [self.joint_pos, self.data.qvel[6 : 6 + N_JOINTS], *rel, *to_cubes,
             [self.pelvis_height]]
        ).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return 2 * N_JOINTS + 9 + 9 + 1

    # ---------- lifecycle ----------

    def reset(self, config_idx: int | None = None):
        self.sim.reset()                 # their reset; rewinds the clock
        self.sim.randomize_cubes()       # their cube schedule
        if config_idx is not None:
            self.sim._episode_count = config_idx
            self.sim.randomize_cubes()
        self.data.qpos[7 : 7 + N_JOINTS] = INIT_POSE

        self.sim_time = 0.0
        self.wbc.reset(sim_time=0.0)
        self.wbc.policy.activate_policy()
        self.episode += 1
        self.chunks = 0

        self._run(pose_as_chunk(INIT_POSE), steps=200)
        self.base_noise = self.policy.base_noise(self.rng)
        self.prev_score = self.placements()
        self.prev_progress = self.progress_distance()
        return self.observation()

    @staticmethod
    def _slice_chunk(chunk: dict, t: int) -> dict:
        return {k: v[t : t + 1][None, ...] for k, v in chunk.items()}

    def _run(self, chunk_slice: dict, steps: int):
        import mujoco

        for i in range(steps):
            if i % WBC_DECIMATION == 0:
                self.wbc.step(self.data, chunk_slice, self.sim_time, self.cfg.env.control_hz)
            self.data.ctrl[:N_JOINTS] = self.wbc.torques(self.data)
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.model.opt.timestep
            if self.viewer is not None and i % 4 == 0:
                self.viewer.sync()
                if self.realtime:
                    time.sleep(self.model.opt.timestep * 4)

    # ---------- step ----------

    def step(self, action: np.ndarray):
        steer = np.clip(action, -1.0, 1.0) * self.cfg.groot.noise_scale
        seed = self.policy.build_seed(self.base_noise, steer)

        obs_in = self.policy.observe(self.joint_pos, self.render())
        chunk = self.policy.chunk(obs_in, seed)

        for k in range(self.cfg.env.execute_steps):
            self._run(self._slice_chunk(chunk, k), self.sim_per_ctrl)

        self.chunks += 1
        score = self.placements()
        r = self.cfg.reward
        reward = ((score["correct"] - self.prev_score["correct"]) * r.correct_placement
                  + (score["wrong"] - self.prev_score["wrong"]) * r.wrong_target)
        self.prev_score = score

        progress = self.progress_distance()
        shaped = np.clip(
            (self.prev_progress - progress) * r.approach_coeff, -r.approach_cap, r.approach_cap
        )
        self.prev_progress = progress
        reward += float(shaped)

        fell = self.pelvis_height < r.fall_height
        if fell:
            reward += r.fell
        info = {"correct": score["correct"], "wrong": score["wrong"], "fell": fell,
                "progress": progress, "shaped": float(shaped)}
        done = fell or score["correct"] == self.cfg.env.n_cubes
        truncated = self.chunks >= self.cfg.env.max_chunks
        return self.observation(), float(reward), done, truncated, info
