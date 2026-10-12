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
from dsrl.rewards import RewardFn
from dsrl.sim.wbc_adapter import WbcAdapter, N_JOINTS

FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")

HEIGHT_CMD = 0.74
WBC_DECIMATION = 4
COLORS = ("red", "green", "blue")

# Episode-start base pose of the healthy reference stack (x, y, yaw), measured
# over n=14 recorded episode starts (ref30b, 2026-08-17): x -0.067±0.012,
# y +0.011±0.010, yaw -0.114±0.088. Left free, our settle drifts to y≈+0.05,
# yaw≈+0.05 — a ~4 cm + ~9° lateral bias that displaces every reach the VLA
# makes (it emits body-relative joint targets) and cost ~50 pts of per-cube
# success before it was pinned (11% → 69%, see EXPERIMENTS.md F15–F17).
BASE_CANON_XY_YAW = (-0.067, 0.011, -0.114)

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
            steerable=cfg.groot.steerable_groups, steer_rows=cfg.groot.steer_rows,
            instruction=cfg.env.instruction,
        )
        self.rng = np.random.default_rng(seed)

        self.cube_bid = [self.model.body(f"cube_{c}").id for c in COLORS]
        self.cube_qadr = [self.model.jnt_qposadr[self.model.body(f"cube_{c}").jntadr[0]]
                          for c in COLORS]
        self.cube_vadr = [self.model.jnt_dofadr[self.model.body(f"cube_{c}").jntadr[0]]
                          for c in COLORS]

        # Same indices the ROS-stack task server uses, so RewardFn sees the same
        # state dict in-process that it sees against the reference stack.
        finger_bid = [self.model.body(f"right_hand_{n}_link").id for n in FINGERS]
        self._finger_geoms = {g for g in range(self.model.ngeom)
                              if self.model.geom_bodyid[g] in finger_bid}
        self._cube_geoms = [self.model.geom(f"cube_{c}_geom").id for c in COLORS]
        self.reward_fn = RewardFn(cfg.reward)
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
        # Downscaled ego view for PixelSAC (updated on reset and after every
        # step when sac.use_pixels; None otherwise). uint8 (pixel_size,)*2 x 3.
        self.last_pixels = None

    def _update_pixels(self):
        if not self.cfg.sac.use_pixels:
            return
        import cv2

        s = self.cfg.sac.pixel_size
        self.last_pixels = cv2.resize(self.render(), (s, s), interpolation=cv2.INTER_AREA)

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

    def _finger_contacts(self):
        hits = set()
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self._finger_geoms and b in self._cube_geoms:
                    hits.add(self._cube_geoms.index(b))
        return sorted(hits)

    def task_state(self) -> dict:
        """The state dict RewardFn consumes — same fields as the ROS-stack
        task_server serves, built from the in-process mj_data."""
        return {
            "sim_time": float(self.data.time),
            "cube_pos": np.array([self.data.xpos[b] for b in self.cube_bid], dtype=np.float32),
            "cube_quat": np.array([self.data.xquat[b] for b in self.cube_bid], dtype=np.float32),
            "cube_vel": np.array([self.data.qvel[a:a + 3] for a in self.cube_vadr], dtype=np.float32),
            "basket_pos": np.array([self.data.xpos[b] for b in self.basket_bid], dtype=np.float32),
            "basket_quat": np.array([self.data.xquat[b] for b in self.basket_bid], dtype=np.float32),
            "hand_pos": np.asarray(self.data.xpos[self.hand_bid], dtype=np.float32),
            "contacts": self._finger_contacts(),
            "joint_acc": np.asarray(self.data.qacc[6:6 + N_JOINTS], dtype=np.float32),
            "pelvis_height": self.pelvis_height,
            "fallen": bool(self.pelvis_height < self.cfg.reward.fall_height),
        }

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
    
    def _snap_base(self, keep_qvel: bool = False):
        """Pin base x, y, yaw to the reference's canonical episode-start pose,
        keeping the settled z, roll, pitch and all joints."""
        import mujoco

        q = self.data.qpos
        x0, y0, yaw0 = BASE_CANON_XY_YAW
        q[0], q[1] = x0, y0
        w, x, y, z = q[3:7]
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        cy, sy = np.cos(yaw0 / 2), np.sin(yaw0 / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        q[3] = cr * cp * cy + sr * sp * sy
        q[4] = sr * cp * cy - cr * sp * sy
        q[5] = cr * sp * cy + sr * cp * sy
        q[6] = cr * cp * sy - sr * sp * cy
        if not keep_qvel:
            self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, config_idx: int | None = None):
        import mujoco

        self.sim.reset()                 # their reset; rewinds the clock
        self.sim.randomize_cubes()       # their cube schedule
        if config_idx is not None:
            self.sim._episode_count = config_idx
            self.sim.randomize_cubes()
        self.data.qpos[7 : 7 + N_JOINTS] = INIT_POSE
        # Snapshot the cube placement chosen above; it is re-applied once the
        # robot is still, so episodes start with pristine cube poses the way the
        # reference stack's do (the settle otherwise nudges them ~1.5 cm).
        cube_qpos0 = [self.data.qpos[a : a + 7].copy() for a in self.cube_qadr]

        self.sim_time = 0.0
        self.wbc.reset(sim_time=0.0)
        self.wbc.policy.activate_policy()
        self.episode += 1
        self.chunks = 0

        # Band holds the robot while the WBC stabilises, then is released -- the
        # RUNBOOK's "press 9 in the viewer". Left engaged, the robot hangs ~22 cm
        # high and the cubes are out of reach; never engaged, it sags into a crouch
        # before the WBC has authority.
        self._run(pose_as_chunk(INIT_POSE), steps=200)
        band = getattr(self.sim, "elastic_band", None)
        if band is not None and band.enable:
            band.enable = False
        # The transient takes ~400 steps past band release to die out (pelvis
        # 0.732 → 0.7465 = reference); 200 was a still-sinking snapshot. The
        # settle also lets the base wander, so it is snapped back to the
        # reference pose, re-grounded, and pinned once more at episode start.
        self._run(pose_as_chunk(INIT_POSE), steps=600)
        self._snap_base()
        self._run(pose_as_chunk(INIT_POSE), steps=200)
        self._snap_base(keep_qvel=True)

        for a, v, q0 in zip(self.cube_qadr, self.cube_vadr, cube_qpos0):
            self.data.qpos[a : a + 7] = q0
            self.data.qvel[v : v + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.base_noise = self.policy.base_noise(self.rng)
        self.prev_score = self.placements()
        self.prev_progress = self.progress_distance()
        self.reward_fn.reset(self.task_state())
        self._update_pixels()
        return self.observation()

    @staticmethod
    def _slice_chunk(chunk: dict, t: int) -> dict:
        return {k: v[t : t + 1][None, ...] for k, v in chunk.items()}

    def _apply_elastic_band(self, mujoco):
        """Their sim_step applies this every step; stepping MuJoCo directly skips it.

        ENABLE_ELASTIC_BAND is True for this scene, so the deployed sim hangs the
        robot from a virtual spring. Without it the legs carry the full weight and
        the robot settles ~22 cm lower, out of the workspace the VLA was trained in.
        """
        band = getattr(self.sim, "elastic_band", None)
        if not band:
            return
        link = self.sim.band_attached_link
        # xquat is only valid once forward kinematics have run. Their loop always
        # steps before reaching here; reset() assigns qpos and calls this first.
        if np.linalg.norm(self.data.xquat[link]) < 1e-9:
            mujoco.mj_forward(self.model, self.data)
        if band.enable and getattr(self.sim, "use_floating_root_link", True):
            pose = np.concatenate(
                [self.data.xpos[link], self.data.xquat[link], np.zeros(6)]
            )
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, link, pose[7:13], 0
            )
            pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
            self.data.xfrc_applied[link] = band.Advance(pose)
        else:
            self.data.xfrc_applied[link] = np.zeros(6)

    def _run(self, chunk_slice: dict, steps: int):
        import mujoco

        # set_goal is the VLA-rate call: one waypoint per slice. Re-posting it every
        # WBC tick would move the target to now + 1/freq each time, so the
        # interpolator chases a receding waypoint and the arm crawls.
        self.wbc.set_goal(chunk_slice, self.sim_time, self.cfg.env.control_hz)

        for i in range(steps):
            if i % WBC_DECIMATION == 0:
                self.wbc.tick(self.data, self.sim_time)
            self.data.ctrl[:N_JOINTS] = self.wbc.torques(self.data)
            self._apply_elastic_band(mujoco)
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.model.opt.timestep
            if self.viewer is not None and i % 4 == 0:
                self.viewer.sync()
                if self.realtime:
                    time.sleep(self.model.opt.timestep * 4)

    # ---------- step ----------

    def step(self, action: np.ndarray):
        if self.cfg.groot.resample_unsteered:
            self.base_noise = self.policy.base_noise(self.rng)
        steer = np.clip(action, -1.0, 1.0) * self.cfg.groot.noise_scale
        seed = self.policy.build_seed(self.base_noise, steer,
                                      residual=self.cfg.groot.steer_residual)

        obs_in = self.policy.observe(self.joint_pos, self.render())
        chunk = self.policy.chunk(obs_in, seed)

        for k in range(self.cfg.env.execute_steps):
            self._run(self._slice_chunk(chunk, k), self.sim_per_ctrl)

        self.chunks += 1
        # RewardFn is the audited reward (once-per-cube milestones, revocation,
        # slip/drop/disturb/tip penalties) — the same one every reference-stack
        # run trained against. Evaluated on the post-chunk state, like StackEnv.
        reward, info = self.reward_fn(self.task_state())
        r = self.cfg.reward

        progress = self.progress_distance()
        shaped = np.clip(
            (self.prev_progress - progress) * r.approach_coeff, -r.approach_cap, r.approach_cap
        )
        self.prev_progress = progress
        reward += float(shaped)
        info["progress"] = progress
        info["shaped"] = float(shaped)

        fell = info["fell"]
        done = fell or info["correct"] == self.cfg.env.n_cubes
        # Nothing left to do: every cube is placed (rightly or wrongly) or has
        # left the table for good. Ending here saves dead chunks at the tail.
        if self.cfg.env.early_terminate and info.get("resolved", 0) >= self.cfg.env.n_cubes:
            done = True
        truncated = self.chunks >= self.cfg.env.max_chunks
        self._update_pixels()
        return self.observation(), float(reward), done, truncated, info
