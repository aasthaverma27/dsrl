"""Drives the shipped WBC policy in-process — no reimplementation, no DDS/ROS.

The controller's control/policy layer (decoupled_wbc, from NVIDIA's
GR00T-WholeBodyControl) is transport-free (numpy + onnxruntime only); the
rclpy/DDS machinery lives in run_g1_control_loop.py, which is the deployment
wrapper, not the controller. So the real G1DecoupledWholeBodyPolicy runs here
directly, driven by our MuJoCo clock instead of wall time.

Their robot_model already indexes the same 43-dim joint vector our scene uses
(lower_body 0..14, left_arm 15..21, right_arm 29..35), so no remapping is needed.
"""

import os
from pathlib import Path

# NVIDIA GR00T-WholeBodyControl checkout (scripts/setup.sh puts it in external/).
WBC_ROOT = Path(os.environ.get(
    "DSRL_WBC_ROOT", Path(__file__).resolve().parents[2] / "external" / "GR00T-WholeBodyControl"))
import time as time_module

import numpy as np
import yaml

from dsrl.sim.goal_builder import policy_action_to_control_goal

SIM_CFG = WBC_ROOT / "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"
DEPLOY_CFG = WBC_ROOT / "decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml"

N_JOINTS = 43
N_BODY_MOTORS = 29

# Hand joints are not in MOTOR_KP (that covers the 29 body motors). Their real gains
# are the ones the controller's Dex3 sender puts in every HandCmd -- command_sender.py
# sets kp=[2,1,1,1,1,1,1], kd=[0.5,0.2,...] -- and gear_sonic's compute_hand_torques
# uses those, not a sim-side constant. The previous kp=100/kd=0.5 guess was ~100x too
# stiff: fingers drove to the commanded angle instead of complying, and the contact
# solver ejected the cube rather than letting it be held.
HAND_KP = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
HAND_KD = np.array([0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
LEFT_HAND = slice(22, 29)
RIGHT_HAND = slice(36, 43)



class WbcAdapter:
    """Thin bridge: MuJoCo state -> their policy -> joint targets -> MuJoCo torques."""

    def __init__(self, upper_body_max_joint_speed: float = 1000.0):
        from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

        self.robot_model = instantiate_g1_robot_model(
            waist_location="lower_body", high_elbow_pose=False
        )
        # Keys the factory needs that live on their CLI dataclass, not the yaml.
        # Same values run_g1_control_loop.py defaults to; paths resolve relative to
        # sim2mujoco/resources/robots/g1.
        wbc_config = yaml.safe_load(DEPLOY_CFG.read_text())
        wbc_config["model_path"] = (
            "policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx"
        )
        wbc_config["upper_body_max_joint_speed"] = upper_body_max_joint_speed
        wbc_config.setdefault("enable_waist", False)
        wbc_config.setdefault("with_hands", True)
        wbc_config.setdefault("high_elbow_pose", False)
        wbc_config.setdefault("enable_gravity_compensation", False)
        self._wbc_config = wbc_config
        self._get_wbc_policy = get_wbc_policy

        sim_cfg = yaml.safe_load(SIM_CFG.read_text())
        # Gains the gear_sonic bridge applies to LowCmd targets; reused here so the
        # in-process env tracks the same way the deployed sim does.
        self.motor_kp = np.asarray(sim_cfg["MOTOR_KP"], dtype=np.float64)
        self.motor_kd = np.asarray(sim_cfg["MOTOR_KD"], dtype=np.float64)
        self.effort_limit = np.asarray(sim_cfg["motor_effort_limit_list"], dtype=np.float64)
        self.body_idx = np.asarray(self.robot_model.get_joint_group_indices("body"))
        self.policy = None
        self._build_goal = policy_action_to_control_goal

        self.reset()

    def _make_hand_smoother(self, freq: int = 20):
        from decoupled_wbc.control.utils.hand_grasp_smoother import HandGraspSmoother

        return HandGraspSmoother(dt=1.0 / freq)

    def reset(self, sim_time: float = 0.0):
        """Reset in place. Their deploy loop builds the policy once; rebuilding it
        per episode reloads both ONNX files from disk and reprints their banner."""
        if getattr(self, "policy", None) is None:
            self.policy = self._get_wbc_policy(
                "g1", self.robot_model, self._wbc_config, init_time=sim_time
            )
        else:
            lower = self.policy.lower_body_policy
            lower.obs_history.clear()
            lower.obs_buffer[:] = 0.0
            lower.action[:] = 0.0
            lower.target_dof_pos = lower.config["default_angles"].copy()
            lower.counter = 0
            self.policy.upper_body_policy.reset(init_time=sim_time)

        self.hand_smoother = self._make_hand_smoother()
        self.last_q = None

    # ---------- their contract ----------

    @staticmethod
    def observation_from(data) -> dict:
        """The dict G1Env hands the policy, built from MuJoCo instead of DDS."""
        return {
            "q": data.qpos[7 : 7 + N_JOINTS].copy(),
            "dq": data.qvel[6 : 6 + N_JOINTS].copy(),
            "floating_base_pose": data.qpos[:7].copy(),
            "floating_base_vel": data.qvel[:6].copy(),
        }

    def set_goal(self, chunk_slice: dict, sim_time: float, freq: int):
        """Publish one waypoint — call at the VLA rate (20 Hz), not per WBC tick.

        Their loop only calls set_goal when a new message arrives. Calling it every
        tick re-posts the target at now + 1/freq each time, so the interpolator
        chases a receding waypoint and the arm crawls.
        """
        goal = self._build_goal(chunk_slice, sim_time, freq)
        goal["interpolation_garbage_collection_time"] = sim_time - 2 * (1 / freq)
        self.policy.set_goal(goal)

    def tick(self, data, sim_time: float):
        """One WBC control tick (50 Hz): observe, act, smooth."""
        self.policy.set_observation(self.observation_from(data))
        action = self.policy.get_action(time=sim_time)
        if self.hand_smoother is not None:
            self.hand_smoother.smooth(action["q"])
        self.last_q = np.asarray(action["q"], dtype=np.float64)
        return self.last_q

    def torques(self, data) -> np.ndarray:
        """PD to the policy's joint targets, using the sim bridge's own gains."""
        q = data.qpos[7 : 7 + N_JOINTS]
        dq = data.qvel[6 : 6 + N_JOINTS]
        target = self.last_q if self.last_q is not None else q
        # Hand channel rotation. Their pipeline packs HandCmd motors as
        # q[[40,41,42,36,37,38,39]] (robot_model.get_hand_actuated_joint_indices)
        # and gear_sonic's bridge applies motors to mujoco hand joints in mujoco
        # order -- a net rotation the training data was collected through, so the
        # checkpoint's hand channels only make sense with it. Verified empirically:
        # their achieved joints equal cmd[[4,5,6,0,1,2,3]] including limit clamps.
        target = target.copy()
        target[LEFT_HAND] = target[[26, 27, 28, 22, 23, 24, 25]]
        target[RIGHT_HAND] = target[[40, 41, 42, 36, 37, 38, 39]]
        kp = np.empty(N_JOINTS)
        kd = np.empty(N_JOINTS)
        kp[self.body_idx] = self.motor_kp[:N_BODY_MOTORS]
        kd[self.body_idx] = self.motor_kd[:N_BODY_MOTORS]
        for sl in (LEFT_HAND, RIGHT_HAND):
            kp[sl], kd[sl] = HAND_KP, HAND_KD
        tau = (target - q) * kp - dq * kd
        # Their sim clips before writing ctrl; without this a finger rated 0.7 Nm
        # gets whatever the PD asks for.
        return np.clip(tau, -self.effort_limit, self.effort_limit)
