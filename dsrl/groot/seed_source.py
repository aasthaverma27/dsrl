"""Hook 3: supply the diffusion seed to gr00t_wbc's inference client.

DSRL chooses the seed; their client is the process that calls the policy server.
The client asks this module for the current seed and passes it through as
options={"dsrl_noise": ...}, which PolicyClient already forwards over ZMQ and the
two-line Isaac-GR00T patch consumes.

A missing or stale seed yields None, so their client behaves exactly as shipped
(the model samples its own noise) whenever DSRL is not driving.
"""

import os
import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

DEFAULT_PORT = int(os.environ.get("DSRL_SEED_PORT", 5598))
ACK_PORT = int(os.environ.get("DSRL_ACK_PORT", 5597))
# A seed older than this is ignored: DSRL is only steering while it is publishing.
# Two chunks (0.8 s each) plus slack.
SEED_TTL = float(os.environ.get("DSRL_SEED_TTL", 2.5))
_sub = None
_ack = None
_latest = None
_latest_at = 0.0

def _socket():
    """PULL, not SUB: PUB/SUB silently drops anything published before the
    subscriber attaches, which would lose seeds whenever DSRL starts first."""
    global _sub
    if _sub is None:
        ctx = zmq.Context.instance()
        _sub = ctx.socket(zmq.PULL)
        _sub.connect(f"tcp://localhost:{DEFAULT_PORT}")
    return _sub

def _ack_socket():
    global _ack
    if _ack is None:
        ctx = zmq.Context.instance()
        _ack = ctx.socket(zmq.PUSH)
        _ack.connect(f"tcp://localhost:{ACK_PORT}")
    return _ack

def current_options():
    """Return the options dict for PolicyClient.get_action, or None.

    Also acks the seed id actually consumed. Their client queries the policy on
    its own schedule, so without an ack DSRL cannot know which seed produced which
    chunk, and the critic would be trained on misaligned transitions.
    """
    global _latest, _latest_at
    sock = _socket()
    try:
        while True:
            msg = sock.recv(flags=zmq.NOBLOCK)
            _latest = msgpack.unpackb(msg, object_hook=mnp.decode, raw=False)
            _latest_at = time.monotonic()
    except zmq.Again:
        pass
    # Expire stale seeds. Without this the last seed is reused for every chunk
    # forever once DSRL stops publishing, so the "frozen policy" baseline would
    # silently run on a stale steered seed instead of the model's own noise.
    if _latest is not None and time.monotonic() - _latest_at > SEED_TTL:
        _latest = None
    if _latest is None:
        return None
    seed = np.asarray(_latest["seed"], dtype=np.float32)
    try:
        _ack_socket().send(msgpack.packb({"step": int(_latest.get("step", -1))}))
    except Exception:
        pass
    return {"dsrl_noise": seed}


class SeedPublisher:
    """DSRL side: publish a seed and wait for the client to acknowledge using it."""

    def __init__(self, port: int = DEFAULT_PORT, ack_port: int = ACK_PORT):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.bind(f"tcp://0.0.0.0:{port}")
        self.ack = ctx.socket(zmq.PULL)
        self.ack.bind(f"tcp://0.0.0.0:{ack_port}")

    def send(self, seed: np.ndarray, step: int = 0):
        self.sock.send(
            msgpack.packb(
                {"seed": np.asarray(seed, dtype=np.float32), "step": step},
                default=mnp.encode,
            )
        )

    def wait_ack(self, step: int, timeout: float = 3.0) -> bool:
        """Block until the client reports consuming this step's seed."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if self.ack.poll(int(remaining * 1000)) == 0:
                return False
            msg = msgpack.unpackb(self.ack.recv(), raw=False)
            if msg.get("step", -1) >= step:
                return True
        return False

    def wait_next_ack(self, timeout: float = 3.0) -> bool:
        """Block for one further ack, whatever id it carries.

        The client re-acks the seed it still holds on every query, so a later id
        never arrives until DSRL publishes again. Counting acks is what marks the
        chunk boundary.
        """
        return self.ack.poll(int(timeout * 1000)) != 0 and bool(self.ack.recv())

    def drain_acks(self):
        while self.ack.poll(0):
            self.ack.recv()
