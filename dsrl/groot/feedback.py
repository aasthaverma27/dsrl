"""Hook 4: return what the policy produced, so DSRL can learn in action space.

DSRL-NA (paper Algorithm 1) needs two things our seed/ack channel cannot supply:

  Q^A is a critic over ENVIRONMENT actions, so the buffer must hold the action chunk
  the policy actually produced -- their collect_rollouts stores `action`, not `noise`.

  The Q^W distillation samples a state from the buffer, draws fresh w ~ N(0,I), and
  computes a = pi_dp(s, w). That means re-denoising at replayed states. Re-running the
  2B VLM per replayed state is not affordable, but the backbone features are already
  computed once per real step, so storing them lets the action head alone be re-run
  locally -- verified feasible: the head loads in ~2.6GB and denoises off a synthetic
  feature map.

The client pushes one record per policy query; DSRL pulls them. PUSH/PULL rather than
PUB/SUB for the same reason as the seed channel: PUB silently drops anything sent
before the subscriber attaches.
"""

import os
import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

PORT = int(os.environ.get("DSRL_FEEDBACK_PORT", 5596))
_push = None


def publish(step: int, action_chunk, info: dict, extra_noise=None) -> None:
    """Called in their client after get_action. Never raises: a feedback failure must
    not take down the control loop, since the loop is driving a humanoid.

    `extra_noise` is echoed back so DSRL can pair each candidate seed with the action
    it produced. Pairing here rather than by index avoids any ordering assumption
    between the seed channel and this one.
    """
    global _push
    try:
        if _push is None:
            ctx = zmq.Context.instance()
            _push = ctx.socket(zmq.PUSH)
            _push.setsockopt(zmq.SNDHWM, 64)      # bounded: drop rather than stall
            _push.setsockopt(zmq.LINGER, 0)
            _push.connect(f"tcp://localhost:{PORT}")
        rec = {"step": int(step), "t": time.monotonic()}
        for key in ("action_pred", "backbone_features", "state_features",
                    "dsrl_candidate_actions"):
            if key in info:
                rec[key] = np.asarray(info[key], dtype=np.float16)
        if extra_noise is not None:
            rec["candidate_noise"] = np.asarray(extra_noise, dtype=np.float16)
        _push.send(msgpack.packb(rec, default=mnp.encode), flags=zmq.NOBLOCK)
    except Exception:
        pass


class FeedbackReceiver:
    """DSRL side. Binds and drains; keeps only the most recent `window` records.

    A rolling window, not the full run: backbone features are ~256 KB per step at
    float16, so 20,000 steps would be 5 GB. The distillation only needs states it can
    still sample, so recent states are what matter.
    """

    def __init__(self, port: int = PORT, window: int = 4000):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVHWM, 256)
        self.sock.bind(f"tcp://0.0.0.0:{port}")
        self.window = window
        self.records: dict[int, dict] = {}
        self.order: list[int] = []

    def drain(self) -> int:
        """Pull everything queued. Returns how many records arrived."""
        n = 0
        while True:
            try:
                rec = msgpack.unpackb(
                    self.sock.recv(flags=zmq.NOBLOCK), object_hook=mnp.decode, raw=False
                )
            except zmq.Again:
                break
            step = int(rec.get("step", -1))
            if step not in self.records:
                self.order.append(step)
            self.records[step] = rec
            n += 1
        while len(self.order) > self.window:
            self.records.pop(self.order.pop(0), None)
        return n

    def latest(self) -> dict | None:
        return self.records.get(self.order[-1]) if self.order else None

    def get(self, step: int) -> dict | None:
        return self.records.get(step)

    def __len__(self) -> int:
        return len(self.order)
