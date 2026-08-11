import numpy as np

BASKET_INNER = 0.05
# Wall centre 0.055 + half-thickness 0.005: the basket's true outer half-extent.
# Used to decide "released above a basket" — a looser radius swallows most of the
# 0.13 m table-to-basket path and suppresses real mid-transport drops.
BASKET_OUTER = 0.06
BASKET_FLOOR_Z = 0.010
CUBE_HALF = 0.035
N_CUBES = 3
TABLE_Z = 0.795
LIFT_MARGIN = 0.02
DISTURB_MOVE = 0.02

def placed_mask(state: dict) -> np.ndarray:
    """(3, 3) bool: cube i is resting in basket j."""
    cube, basket = state["cube_pos"], state["basket_pos"]
    out = np.zeros((N_CUBES, N_CUBES), dtype=bool)
    for i in range(N_CUBES):
        for j in range(N_CUBES):
            rest_z = basket[j][2] + BASKET_FLOOR_Z + CUBE_HALF
            out[i, j] = (
                np.abs(cube[i][:2] - basket[j][:2]).max() < BASKET_INNER
                and abs(cube[i][2] - rest_z) < 2 * CUBE_HALF
            )
    return out


def tipped_mask(quats: np.ndarray, tip_angle_deg: float) -> np.ndarray:
    """Bodies whose local z-axis has fallen more than tip_angle_deg from world up."""
    x, y = quats[:, 1], quats[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return up_z < np.cos(np.deg2rad(tip_angle_deg))

class RewardFn:
    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self, state: dict) -> None:
        self.prev_placed = placed_mask(state)
        self.prev_cube_pos = state["cube_pos"].copy()
        self.prev_contacts = set(state["contacts"])
        self.prev_basket_tipped = tipped_mask(state["basket_quat"], self.cfg.tip_angle_deg)
        self.lifted = set()
        self.target = None
        self.settle_count = np.zeros(N_CUBES, dtype=int)
        # Milestones already reached, so each pays at most once per cube per episode.
        # Without this a policy could farm them by repeating the same motion.
        self.paid_contact = set()
        self.paid_lifted = set()
        self.paid_over = set()

    def __call__(self, state: dict) -> tuple[float, dict]:
        cfg = self.cfg
        placed = placed_mask(state)
        contacts = set(state["contacts"])
        cube_pos, cube_vel = state["cube_pos"], state["cube_vel"]
        events = {}
        reward = 0.0

        newly_placed = placed & ~self.prev_placed
        for i in range(N_CUBES):
            for j in range(N_CUBES):
                if newly_placed[i, j]:
                    reward += cfg.correct_placement if i == j else cfg.wrong_target
                    events["correct_placement" if i == j else "wrong_target"] = i

        if contacts:
            self.target = max(contacts)
        for i in contacts:
            if cube_pos[i][2] > TABLE_Z + LIFT_MARGIN:
                self.lifted.add(i)

        # Milestones: touched -> lifted -> carried over the right basket -> placed.
        # Each is irreversible progress toward a placement, so paying them turns a
        # ~3-event episode into a ~12-event one without changing what "best" means.
        for i in contacts:
            if i not in self.paid_contact:
                self.paid_contact.add(i)
                reward += cfg.first_contact
                events["first_contact"] = i
        for i in self.lifted:
            if i not in self.paid_lifted:
                self.paid_lifted.add(i)
                reward += cfg.lifted
                events["lifted"] = i
        for i in self.lifted - self.paid_over:
            # over its OWN basket, not any basket: this is the signal that the cube
            # is being taken to the right place, which is the decision we want shaped
            if float(np.abs(cube_pos[i][:2] - state["basket_pos"][i][:2]).max()) < BASKET_OUTER:
                self.paid_over.add(i)
                reward += cfg.over_correct_basket
                events["over_correct_basket"] = i

        # A cube that was lifted and is no longer held: slip if still falling,
        # dropped once it has settled outside every basket. Releasing above a basket
        # also reads as "falling and not yet placed", so that case is excluded —
        # otherwise every successful placement is charged a slip first.
        for i in sorted(self.lifted):
            if i in contacts:
                continue
            in_any_basket = bool(placed[i].any())
            descending = cube_vel[i][2] < -cfg.settle_speed
            over_basket = min(
                float(np.abs(cube_pos[i][:2] - state["basket_pos"][j][:2]).max())
                for j in range(N_CUBES)
            ) < BASKET_OUTER
            if descending and not in_any_basket and not over_basket:
                reward += cfg.grasp_slip
                events["grasp_slip"] = i
                self.lifted.discard(i)
                continue
            if np.linalg.norm(cube_vel[i]) < cfg.settle_speed:
                self.settle_count[i] += 1
                if self.settle_count[i] >= cfg.settle_steps:
                    if not in_any_basket:
                        reward += cfg.dropped
                        events["dropped"] = i
                    self.lifted.discard(i)
                    self.settle_count[i] = 0
            else:
                self.settle_count[i] = 0

        moved = np.linalg.norm(cube_pos - self.prev_cube_pos, axis=1)
        for i in range(N_CUBES):
            if i == self.target or i in contacts:
                continue
            if moved[i] > DISTURB_MOVE and not placed[i].any():
                reward += cfg.disturb
                events["disturb"] = i

        basket_tipped = tipped_mask(state["basket_quat"], cfg.tip_angle_deg)
        newly_tipped = basket_tipped & ~self.prev_basket_tipped
        if newly_tipped.any():
            reward += cfg.target_tipped * int(newly_tipped.sum())
            events["target_tipped"] = int(np.argmax(newly_tipped))

        jerk = float(np.linalg.norm(state["joint_acc"]))
        reward += cfg.jerk_coeff * jerk

        if state["fallen"]:
            reward += cfg.fell

        self.prev_placed = placed
        self.prev_cube_pos = cube_pos.copy()
        self.prev_contacts = contacts
        self.prev_basket_tipped = basket_tipped

        correct = int(sum(placed[i, i] for i in range(N_CUBES)))
        # A cube far below the table is unrecoverable, so an episode where every cube
        # is either placed or gone has nothing left to do. Threshold is generous
        # (0.15 m below the surface) so a cube bouncing on the table never counts.
        off_table = [
            i for i in range(N_CUBES)
            if cube_pos[i][2] < TABLE_Z - self.cfg.off_table_drop and not placed[i].any()
        ]
        return float(reward), {
            "correct": correct,
            "wrong": int(placed.sum() - correct),
            "fell": bool(state["fallen"]),
            "jerk": jerk,
            "events": events,
            "off_table": off_table,
            "resolved": int(placed.any(axis=1).sum()) + len(off_table),
        }
