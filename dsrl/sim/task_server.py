"""DSRL's hook into gear_sonic's sim: serve task state, accept resets.

Their sim publishes robot state (DDS) and camera frames (ZMQ) only. DSRL also needs
cube/basket poses to score an episode and a way to reset between episodes, and
neither exists on any channel in either repo.

Runs as a thread inside their sim process. Read-only on mj_data, except for setting
env._reset_requested — their own flag, which their sim_step() already handles.
"""

import threading
import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

DEFAULT_PORT = 5599
COLORS = ("red", "green", "blue")
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
N_JOINTS = 43


def serve(env, port: int = DEFAULT_PORT, cubes_every_episode: bool = True) -> threading.Thread:
    """Start the task-state server on `env` (a gear_sonic DefaultEnv)."""
    if cubes_every_episode:
        # Their schedule holds one arrangement for 16 episodes (good for balanced
        # data collection). RL needs a new arrangement every reset, or the policy
        # overfits a single layout.
        env.episodes_per_config = 1
        print("[dsrl] cube arrangement rotates every episode", flush=True)

    model, data = env.mj_model, env.mj_data
    cube_bid = [model.body(f"cube_{c}").id for c in COLORS]
    # freejoint dof address, so cube linear velocity can be read for slip/drop
    cube_qvel_adr = [model.joint(f"cube_{c}_free").dofadr[0] for c in COLORS]
    basket_bid = [model.body(f"basket_{c}").id for c in COLORS]
    hand_bid = model.body("right_wrist_yaw_link").id
    finger_bid = [model.body(f"right_hand_{n}_link").id for n in FINGERS]
    finger_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] in finger_bid}
    cube_geoms = [model.geom(f"cube_{c}_geom").id for c in COLORS]

    def finger_contacts():
        hits = set()
        for k in range(data.ncon):
            c = data.contact[k]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in finger_geoms and b in cube_geoms:
                    hits.add(cube_geoms.index(b))
        return sorted(hits)

    def state():
        return {
            "sim_time": float(data.time),
            # --- placement / transport ---
            "cube_pos": np.array([data.xpos[b] for b in cube_bid], dtype=np.float32),
            "cube_quat": np.array([data.xquat[b] for b in cube_bid], dtype=np.float32),
            "cube_vel": np.array(
                [data.qvel[a : a + 3] for a in cube_qvel_adr], dtype=np.float32
            ),
            "basket_pos": np.array([data.xpos[b] for b in basket_bid], dtype=np.float32),
            # basket_quat lets a reward detect a tipped basket (target_tipped)
            "basket_quat": np.array([data.xquat[b] for b in basket_bid], dtype=np.float32),
            # --- grasp / slip / drop ---
            "hand_pos": np.asarray(data.xpos[hand_bid], dtype=np.float32),
            "finger_pos": np.array([data.xpos[b] for b in finger_bid], dtype=np.float32),
            "contacts": finger_contacts(),
            "n_contacts": len(finger_contacts()),
            # --- smoothness (jerk_coeff) ---
            "joint_pos": np.asarray(data.qpos[7 : 7 + N_JOINTS], dtype=np.float32),
            "joint_vel": np.asarray(data.qvel[6 : 6 + N_JOINTS], dtype=np.float32),
            "joint_acc": np.asarray(data.qacc[6 : 6 + N_JOINTS], dtype=np.float32),
            # --- balance ---
            "pelvis_height": float(data.qpos[2]),
            "pelvis_quat": np.asarray(data.qpos[3:7], dtype=np.float32),
            "fallen": bool(getattr(env, "fall", False)),
        }

    def stable_state(retries: int = 5):
        """Their sim_step resets mj_data on its own thread, so a read landing
        mid-reset raises. Retry briefly rather than surfacing a transient error."""
        for _ in range(retries):
            try:
                return state()
            except Exception:
                time.sleep(0.02)
        return state()

    def loop():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://0.0.0.0:{port}")
        print(f"[dsrl] task server on {port}", flush=True)
        while True:
            try:
                req = msgpack.unpackb(sock.recv(), object_hook=mnp.decode, raw=False)
                cmd = req.get("cmd") if isinstance(req, dict) else req
                if cmd == "state":
                    reply = stable_state()
                elif cmd == "reset":
                    env._reset_requested = True
                    reply = {"ok": True}
                elif isinstance(cmd, str) and cmd.startswith("layout:"):
                    # their randomize_cubes derives the config from _episode_count,
                    # so setting it pins which arrangement the next reset produces
                    idx = int(cmd.split(":", 1)[1])
                    env._episode_count = idx * env.episodes_per_config
                    reply = {"ok": True, "layout": idx}
                else:
                    reply = {"error": f"unknown cmd {cmd!r}"}
            except Exception as exc:
                reply = {"error": repr(exc)}
            sock.send(msgpack.packb(reply, default=mnp.encode))

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return thread
