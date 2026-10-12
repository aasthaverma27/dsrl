"""Seed-steerable wrapper around the frozen GR00T checkpoint.

DSRL never touches the checkpoint's weights. It chooses the flow-matching initial
noise, which requires the patch in patches/Isaac-GR00T_dsrl_noise.patch
(gr00t_n1d7.py accepting options["dsrl_noise"], gr00t_policy.py forwarding options).
"""

import os
from pathlib import Path

import numpy as np
import torch

# The fine-tuned GR00T N1.7 checkpoint (scripts/setup.sh downloads it here).
CKPT = Path(os.environ.get(
    "DSRL_CHECKPOINT",
    Path(__file__).resolve().parents[2] / "checkpoints/red_green_blue_cube_sorting_in_sim/checkpoint-22000"))
EMBODIMENT = "unitree_g1_full_body_with_waist_height_nav_cmd"

# 43-dim joint vector -> the state groups the policy consumes
STATE_GROUPS = (
    ("left_leg", 0, 6), ("right_leg", 6, 12), ("waist", 12, 15), ("left_arm", 15, 22),
    ("left_hand", 22, 29), ("right_arm", 29, 36), ("right_hand", 36, 43),
)
# served action order, concatenated -> offsets into the live 35-dim action
ACTION_GROUPS = (
    ("left_arm", 7), ("right_arm", 7), ("left_hand", 7), ("right_hand", 7),
    ("waist", 3), ("base_height_command", 1), ("navigate_command", 3),
)
# 43-dim joint indices each action group drives
JOINT_SLICES = {
    "left_arm": (15, 22), "right_arm": (29, 36),
    "left_hand": (22, 29), "right_hand": (36, 43), "waist": (12, 15),
}

def action_offsets():
    out, i = {}, 0
    for name, width in ACTION_GROUPS:
        out[name] = (i, i + width)
        i += width
    return out

class SeededGrootPolicy:
    def __init__(self, steerable=("right_arm", "right_hand"), device="cuda:0",
                 steer_rows: int = 16, instruction: str | None = None):
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(
            embodiment_tag=EMBODIMENT, model_path=str(CKPT), device=device
        )
        cfg = self.policy.model.config
        self.horizon = cfg.action_horizon          # 40, the seed's time extent
        self.seed_dim = cfg.max_action_dim         # 132, padded action width
        self.served = 16                           # action delta_indices [0..15]
        # The fine-tuning dataset's single task string (meta/tasks.jsonl), which
        # EnvConfig.instruction reproduces verbatim.
        if instruction is None:
            from dsrl.config import EnvConfig
            instruction = EnvConfig().instruction
        self.instruction = instruction

        offs = action_offsets()
        self.steerable = tuple(steerable)
        cols = [c for g in self.steerable for c in range(*offs[g])]
        self.seed_cols = np.array(sorted(cols))
        self.seed_rows = np.arange(min(steer_rows, self.served))
        self.action_dim = len(self.seed_rows) * len(self.seed_cols)

    def base_noise(self, rng: np.random.Generator) -> np.ndarray:
        """Fixed noise for the slots DSRL does not steer, so steps stay deterministic."""
        return rng.standard_normal((1, self.horizon, self.seed_dim)).astype(np.float32)

    def build_seed(self, base: np.ndarray, steer: np.ndarray | None,
                   residual: bool = False) -> np.ndarray:
        """steer=None -> the untouched base seed (the no-op).

        residual=True adds the steer to the base-noise slots instead of
        replacing them, so a zero steer is also a no-op; replace semantics
        write the steer values in (zero steer = degenerate 0.0 noise, F18).
        """
        seed = base.copy()
        if steer is not None:
            block = np.asarray(steer, dtype=np.float32).reshape(
                len(self.seed_rows), len(self.seed_cols)
            )
            ix = np.ix_(self.seed_rows, self.seed_cols)
            if residual:
                seed[0][ix] += block
            else:
                seed[0][ix] = block
        return seed

    # Dataset hand convention is [mi0,mi1,in0,in1,th0,th1,th2] (fingers first,
    # thumb last) for BOTH state and action; mujoco is thumb-first. Verified against
    # recorded client obs: identity err 1.5, this rotation err 0.099.
    HAND_STATE_ROT = np.array([3, 4, 5, 6, 0, 1, 2])

    def observe(self, joint_pos: np.ndarray, image: np.ndarray) -> dict:
        state = {}
        for g, a, b in STATE_GROUPS:
            v = joint_pos[a:b]
            if g.endswith("_hand"):
                v = v[self.HAND_STATE_ROT]
            state[g] = np.ascontiguousarray(v[None, None, :], dtype=np.float32)
        return {
            "video": {"ego_view": image[None, None, ...].astype(np.uint8)},
            "state": state,
            "language": {"annotation.human.task_description": [[self.instruction]]},
        }

    def chunk(self, obs: dict, seed: np.ndarray | None = None) -> dict[str, np.ndarray]:
        options = {"dsrl_noise": torch.from_numpy(seed)} if seed is not None else None
        out = self.policy.get_action(obs, options)
        if isinstance(out, tuple):
            out = out[0]
        return {k: np.asarray(v).squeeze(0) for k, v in out.items()}

    def joint_targets(self, chunk: dict[str, np.ndarray], joint_pos: np.ndarray) -> np.ndarray:
        """Expand a chunk into (served, 43) absolute joint targets.

        Groups the policy does not emit (legs) hold their current value; the WBC
        owns them anyway.
        """
        targets = np.tile(joint_pos, (self.served, 1)).astype(np.float32)
        for group, (a, b) in JOINT_SLICES.items():
            if group in chunk:
                targets[:, a:b] = chunk[group][: self.served]
        return targets
