"""Turn one GR00T action step into the whole-body controller's goal.

GR00T N1.7 (embodiment ``unitree_g1_full_body_with_waist_height_nav_cmd``)
predicts per-group joint targets plus navigation and base-height commands; the
deployed controller consumes a single goal dict. This is the same mapping the
deployed inference client uses, kept here as a standalone, numpy-only function
so the in-process environment does not depend on ROS.
"""

import numpy as np


def policy_action_to_control_goal(action: dict, now: float, freq: int) -> dict:
    """
    Convert GR00T policy action to control goal format (matching teleop format)

    Args:
        action: Dict with keys like 'left_arm', 'right_arm', etc.
               Each value is shape (B, T, D) where we extract timestep 0
        now: Current timestamp
        freq: Control frequency

    Returns:
        Control command dict matching teleop format
    """
    control_cmd = {}

    # Extract timestep 0 from each action: (B, T, D) -> (D,)
    def extract_t0(arr):
        """Extract first timestep: (B, T, D) -> (D,)"""
        if isinstance(arr, np.ndarray) and arr.ndim >= 2:
            return arr[0, 0]  # First batch, first timestep
        return arr

    # Upper body pose (43 joints total)
    control_cmd["target_upper_body_pose"] = np.concatenate([
         extract_t0(action.get("left_arm", np.zeros(7))),
          extract_t0(action.get("left_hand", np.zeros(7))),
         extract_t0(action.get("right_arm", np.zeros(7))),
         extract_t0(action.get("right_hand", np.zeros(7))),
    ])

    # Wrist pose (14D)
    control_cmd["wrist_pose"] = np.concatenate([
        extract_t0(action.get("left_wrist_pos", np.zeros(3))),
        # extract_t0(action.get("left_wrist_abs_quat", np.array([1, 0, 0, 0]))),
        # extract_t0(action.get("right_wrist_pos", np.zeros(3))),
        # extract_t0(action.get("right_wrist_abs_quat", np.array([1, 0, 0, 0]))),
    ])

    # Base height
    base_height = extract_t0(action.get("base_height_command", np.array([0.74])))
    control_cmd["base_height_command"] = float(base_height[0] if isinstance(base_height, np.ndarray) else base_height)

    # Navigate command
    nav_cmd = extract_t0(action.get("navigate_command", np.zeros(3)))
    control_cmd["navigate_cmd"] = nav_cmd.tolist() if isinstance(nav_cmd, np.ndarray) else nav_cmd

    # Toggles (matching teleop format)
    control_cmd["toggle_policy_action"] = False
    control_cmd["toggle_data_collection"] = False
    control_cmd["toggle_data_abort"] = False

    # Timing
    control_cmd["timestamp"] = now
    control_cmd["target_time"] = now + (1.0 / freq)

    return control_cmd
