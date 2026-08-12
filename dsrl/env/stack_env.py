"""Gym-style env over the running gear_sonic + gr00t_wbc stack.

Owns no robot control. Each step publishes a diffusion seed to their inference
client, waits for the client to confirm it used that seed, then reads the scene
back and scores it. Everything between is their code.

Requires, all already running:
  task server  :5599   cube/basket state + reset      (dsrl hook in their sim)
  seed  in/out :5598 / :5597                          (dsrl hook in their client)
"""

import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

from dsrl.groot.feedback import FeedbackReceiver
from dsrl.groot.seed_source import SeedPublisher
from dsrl.rewards import RewardFn, placed_mask

TASK_PORT = 5599
N_JOINTS = 43
SEED_SHAPE = (1, 40, 132)
# right_arm and right_hand columns of the served 35-dim action. Steer only the rows
# the client actually executes (cfg.env.execute_steps), because a row that is
# generated and discarded costs a dimension without buying control.
# Authority sweep, 2026-08-11 -- share of executed-action variation governed:
#   rows 0-15 x arm14  224 dims  81.5% control   3.64 control/dim
#   rows 0-3  x arm14   56 dims  75.1% control  13.42 control/dim  <- chosen
#   rows 0-3  x live35 140 dims  81.1% control   5.79 control/dim
STEER_COLS = np.array([7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27])


class StackEnv:
    def __init__(self, cfg, seed: int = 0, task_port: int = TASK_PORT):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        ctx = zmq.Context.instance()
        self.task = ctx.socket(zmq.REQ)
        self.task.setsockopt(zmq.RCVTIMEO, 20000)
        self.task.setsockopt(zmq.LINGER, 0)
        self.task.connect(f"tcp://localhost:{task_port}")
        self.task_port = task_port
        self.seeds = SeedPublisher()
        self.reward_fn = RewardFn(cfg.reward)
        # DSRL-NA: candidate seeds ride along with the real one and are denoised in the
        # same batched pass, so the (noise, action) pairs cost ~7 ms for 16 candidates.
        self.na = cfg.sac.use_noise_aliasing
        self.k_pairs = cfg.sac.na_pairs if self.na else 0
        self.visual = cfg.sac.visual_features
        self.visual_dim = cfg.sac.visual_dim if self.visual else 0
        self.feedback = FeedbackReceiver() if (self.na or self.visual) else None
        # Fixed random projection of the pooled backbone features. Unit-norm rows and
        # an L2-normalised input keep the outputs O(1) with no running statistics to
        # drift. Seeded, so the mapping is identical across runs and checkpoints.
        if self.visual:
            g = np.random.default_rng(12345)
            dim = cfg.groot.backbone_embedding_dim
            P = g.standard_normal((dim, self.visual_dim))
            # Columns scaled to sqrt(dim), not 1. Projecting a unit-norm 2048-dim
            # vector onto unit columns gives components of order 1/sqrt(2048) ~ 0.02,
            # which would sit two orders of magnitude below the rest of the
            # observation (joint angles, contacts, yaw features are all O(1)) and be
            # effectively invisible to the first layer. This puts them at O(1).
            self.visual_proj = (P / np.linalg.norm(P, axis=0, keepdims=True)
                                * np.sqrt(dim)).astype(np.float32)
            self.last_visual = np.zeros(self.visual_dim, dtype=np.float32)

        self.share = cfg.groot.share_across_chunk
        # Only the executed rows are worth steering; see the authority sweep above.
        self.steer_rows = np.arange(cfg.env.execute_steps)
        self.action_dim = len(STEER_COLS) if self.share else len(self.steer_rows) * len(STEER_COLS)
        self.chunk_secs = cfg.env.execute_steps / cfg.env.control_hz
        self.step_index = 0
        self.episode = 0

        # The 5056 unsteered slots. Freezing them removes a hidden variable worth
        # ~45% of the action variation, but binds the run to a single draw -- see
        # GrootConfig.freeze_base_noise for the measurements either way.
        self.freeze_noise = cfg.groot.freeze_base_noise
        self.base_noise = self.rng.standard_normal(SEED_SHAPE).astype(np.float32)
        print(f"[env] action_dim={self.action_dim} = {len(self.steer_rows)} rows x "
              f"{len(STEER_COLS)} cols"
              f"{' (shared across rows)' if self.share else ''}, "
              f"chunk {self.chunk_secs * 1000:.0f}ms, step {2 * self.chunk_secs * 1000:.0f}ms, "
              f"base_noise {'frozen' if self.freeze_noise else 'per-episode'}", flush=True)

    # ---------- task server ----------

    def _call(self, cmd: str, tries: int = 3):
        for _ in range(tries):
            try:
                self.task.send(msgpack.packb({"cmd": cmd}, default=mnp.encode))
                reply = msgpack.unpackb(self.task.recv(), object_hook=mnp.decode, raw=False)
                if isinstance(reply, dict) and "error" in reply:
                    time.sleep(0.1)
                    continue
                return reply
            except zmq.Again:
                # a REQ socket that timed out cannot be reused
                self.task.close()
                ctx = zmq.Context.instance()
                self.task = ctx.socket(zmq.REQ)
                self.task.setsockopt(zmq.RCVTIMEO, 20000)
                self.task.setsockopt(zmq.LINGER, 0)
                self.task.connect(f"tcp://localhost:{self.task_port}")
                time.sleep(0.2)
        raise RuntimeError(f"task server not responding to {cmd!r}")

    # ---------- spaces ----------

    def observation(self, state: dict) -> np.ndarray:
        hand = state["hand_pos"]
        rel = [state["cube_pos"][i] - state["basket_pos"][i] for i in range(3)]
        to_cubes = [state["cube_pos"][i] - hand for i in range(3)]
        contact = np.zeros(3, dtype=np.float32)
        for i in state["contacts"]:
            contact[i] = 1.0
        # Which cubes are already done. Inferable from rel being near zero, but only
        # if the net learns the threshold; explicit costs 3 dims and removes the
        # ambiguity about what is left to do.
        done_flags = placed_mask(state).any(axis=1).astype(np.float32)
        # Cube yaw, with the 4-fold symmetry collapsed. Raw quaternions would make
        # four different values describe the same graspable configuration, so the
        # actor would spend data learning they are equivalent. Only yaw matters for a
        # cube resting on a table: it sets which face the fingers meet.
        q = state["cube_quat"]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        yaw_feat = np.stack([np.cos(4 * yaw), np.sin(4 * yaw)], axis=1).ravel()
        parts = [state["joint_pos"], state["joint_vel"], *rel, *to_cubes,
                 contact, done_flags, yaw_feat, [state["pelvis_height"]]]
        if self.visual:
            parts.append(self.last_visual)
        return np.concatenate(parts).astype(np.float32)

    def _update_visual(self, rec: dict) -> None:
        """Project the backbone features into the observation slot.

        Mean-pooled over tokens then L2-normalised before projecting, so the scale is
        independent of however the backbone's activations happen to be distributed.
        """
        f = np.asarray(rec["backbone_features"], dtype=np.float32)
        v = f.reshape(-1, f.shape[-1]).mean(0)
        n = np.linalg.norm(v)
        self.last_visual = (v / n if n > 1e-6 else v) @ self.visual_proj

    @property
    def obs_dim(self) -> int:
        return 2 * N_JOINTS + 9 + 9 + 3 + 3 + 6 + 1 + self.visual_dim

    # ---------- lifecycle ----------

    def force_layout(self, layout_idx: int):
        """Pin the next reset to one cube arrangement.

        Their randomize_cubes picks the config from _episode_count, so setting it
        before a reset selects the layout. Needed for matched comparisons: without
        it, arrangements advance and any A/B is confounded by layout difficulty.
        """
        self._call(f"layout:{layout_idx}")

    def reset(self):
        self._call("reset")
        time.sleep(self.cfg.env.reset_settle_secs)
        self.seeds.drain_acks()
        state = self._call("state")
        self.reward_fn.reset(state)
        if not self.freeze_noise:
            self.base_noise = self.rng.standard_normal(SEED_SHAPE).astype(np.float32)
        self.chunks = 0
        self.episode += 1
        self.t0 = time.monotonic()
        return self.observation(state)

    def _build_seed(self, action: np.ndarray) -> np.ndarray:
        """Map the actor's box output onto the noise shell the model expects.

        The model denoises from N(0,1); a 224-dim draw from it has norm sqrt(224)
        ~= 15, and in high dimension it lies on that shell almost surely. Scaling
        a tanh output by a constant leaves the shell instead -- uniform(-3,3)^224
        has norm ~26, so every steered seed was ~1.7x longer than any noise the
        frozen policy has seen. Rescaling to the shell keeps the direction, which
        is the only part carrying information, and makes every actor output an
        in-distribution seed.
        """
        # Per-coordinate box, as in the reference: the actor's tanh output in (-1,1)
        # scaled to (-noise_scale, +noise_scale). No norm projection -- that was our
        # own construction and the reference imposes no such constraint.
        steer = np.asarray(action, dtype=np.float32)
        if self.share:
            steer = np.tile(steer, (len(self.steer_rows), 1))
        steer = np.clip(steer, -1.0, 1.0).reshape(len(self.steer_rows), len(STEER_COLS))
        steer = steer * self.cfg.groot.noise_scale
        seed = self.base_noise.copy()
        seed[0][np.ix_(self.steer_rows, STEER_COLS)] = steer
        return seed

    # ---------- noise aliasing ----------

    def env_action(self, chunk: np.ndarray) -> np.ndarray:
        """The executed part of a (40, 132) chunk, flattened: Q^A's action input.

        Only the rows the client executes and the 35 live columns; the rest is either
        discarded before reaching the robot or constant for this embodiment.
        """
        return np.asarray(chunk, dtype=np.float32)[
            np.ix_(self.steer_rows, np.arange(self.cfg.groot.live_action_dim))
        ].ravel()

    @property
    def env_action_dim(self) -> int:
        return len(self.steer_rows) * self.cfg.groot.live_action_dim

    def _collect_pairs(self, cand_noise):
        """Pull this step's feedback and pair each candidate seed with its action.

        Best-effort: the client publishes without blocking, so a missing record just
        means this state is not used for distillation (NaReplayBuffer tracks that).
        """
        self.feedback.drain()
        rec = self.feedback.get(self.step_index)
        if rec is not None and self.visual and "backbone_features" in rec:
            self._update_visual(rec)
        if rec is None or "dsrl_candidate_actions" not in rec:
            return {"na_pairs": None}
        cand_actions = np.asarray(rec["dsrl_candidate_actions"], dtype=np.float32)
        n = min(len(cand_noise), len(cand_actions))
        out = {"na_pairs": (cand_noise[:n],
                            np.stack([self.env_action(a) for a in cand_actions[:n]]))}
        if "action_pred" in rec:
            out["env_action"] = self.env_action(np.asarray(rec["action_pred"],
                                                          dtype=np.float32)[0])
        return out

    # ---------- termination ----------

    def _terminate(self, info: dict, elapsed: float) -> tuple[bool, bool]:
        """done = the episode is genuinely over; truncated = we ran out of budget.

        The distinction matters for bootstrapping: on done the next state's value is
        0 because no further reward is reachable, on truncated it is not.
        "Every cube placed or off the table" is a real ending -- nothing is left to
        earn -- and catching it stops a failed episode burning its remaining 50 s.
        """
        env = self.cfg.env
        done = info["correct"] == env.n_cubes or info["fell"]
        if env.early_terminate and info["resolved"] >= env.n_cubes:
            done = True
            info["ended_early"] = True
        truncated = self.chunks >= env.max_chunks or elapsed > env.max_secs
        return done, truncated

    # ---------- step ----------

    def step(self, action: np.ndarray | None):
        """action=None means publish nothing, so their client samples its own noise —
        the true frozen-policy baseline. Sending zeros instead would write 0.0 into
        224 noise slots, which is a very improbable draw from N(0,1), not 'no steering'."""
        self.step_index += 1
        if action is None:
            time.sleep(2 * self.chunk_secs)
            state = self._call("state")
            reward, info = self.reward_fn(state)
            self.chunks += 1
            info["seed_used"] = False
            elapsed = time.monotonic() - self.t0
            info["elapsed"] = round(elapsed, 1)
            done = info["correct"] == self.cfg.env.n_cubes or info["fell"]
            truncated = self.chunks >= self.cfg.env.max_chunks or elapsed > self.cfg.env.max_secs
            return self.observation(state), reward, done, truncated, info

        cand_noise = None
        if self.na:
            # Candidates are drawn the way the base policy draws noise: clipped
            # standard normal, matching their collect_rollouts. Q^W must learn the
            # value of noise the policy could plausibly receive, and the distillation
            # deliberately covers tails the actor itself cannot reach.
            mag = self.cfg.groot.noise_scale
            z = self.rng.standard_normal((self.k_pairs, self.action_dim))
            cand_noise = np.clip(z, -mag, mag).astype(np.float32) / mag
            extra = np.stack([self._build_seed(c)[0] for c in cand_noise])
        self.seeds.send(self._build_seed(action), step=self.step_index,
                        extra=extra if self.na else None,
                        feat=self.na or self.visual)
        # First ack: the client consumed this seed, so the chunk begins now.
        used = self.seeds.wait_ack(self.step_index, timeout=self.cfg.env.ack_timeout_secs)
        # Second ack: the client queried again, so a full chunk of this seed has
        # executed before we read the scene. One action therefore governs ~1.6 s
        # (two chunks) — their client never pauses, so it re-takes the cached seed
        # before we can publish the next one. Consistent, which is what SAC needs.
        if used:
            self.seeds.wait_next_ack(timeout=self.cfg.env.ack_timeout_secs)
        else:
            time.sleep(self.chunk_secs)
        state = self._call("state")
        reward, info = self.reward_fn(state)

        self.chunks += 1
        info["seed_used"] = used
        elapsed = time.monotonic() - self.t0
        info["elapsed"] = round(elapsed, 1)

        if self.na or self.visual:
            info.update(self._collect_pairs(cand_noise))

        done, truncated = self._terminate(info, elapsed)
        if not used:
            # their client stalled: abort rather than train on a stale transition
            truncated = True
            info["stalled"] = True
        return self.observation(state), reward, done, truncated, info
