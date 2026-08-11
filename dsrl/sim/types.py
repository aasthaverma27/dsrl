"""Scene state. No MuJoCo or torch, so reward logic stays unit-testable."""

import dataclasses
from dataclasses import dataclass, field

Vec3 = tuple[float, float, float]

@dataclass(frozen=True)
class CubeState:
    position: Vec3
    color: str
    speed: float
    grasped: bool

@dataclass(frozen=True)
class TargetState:
    color: str
    center: Vec3
    half_extent: Vec3
    up_axis: Vec3 = (0.0, 0.0, 1.0)

@dataclass(frozen=True)
class SortState:
    cubes: tuple[CubeState, ...]
    targets: tuple[TargetState, ...]
    active_index: int
    settle_counts: tuple[int, ...]
    placed_mask: tuple[bool, ...]
    pelvis_height: float
    chunk_index: int

    @property
    def active_cube(self) -> CubeState:
        return self.cubes[self.active_index]

    def target_for_color(self, color: str) -> TargetState | None:
        for target in self.targets:
            if target.color == color:
                return target
        return None

    def _replace(self, **changes) -> "SortState":
        return dataclasses.replace(self, **changes)

@dataclass(frozen=True)
class RewardResult:
    reward: float
    terms: dict[str, float] = field(default_factory=dict)
    terminal: bool = False
    reason: str = ""
