from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PosteriorConfig:
    dim_h: int = 32
    dim_g: int = 32
    dim_c: int = 32
    sigma_reset: float = 1.0
    q_motion_block: tuple[float, float, float] = (0.01, 0.01, 0.05)
    sigma_min2: float = 1e-4
    sigma_max2: float = 10.0
    point_var_h: float = 4.0
    point_var_g: float = 0.05
    point_var_c: float = 4.0
    visual_var_h: float = 0.5
    visual_var_g: float = 1.0
    visual_var_c: float = 4.0
    n_min_anchors: int = 8
    delta_ref_m: float = 0.005
    epsilon_delta: float = 1e-6
    gamma_min_pc: float = 0.05
    anchor_count_norm: float = 32.0
    force_active_gate: bool = False
    point_radius_min_m: float = 0.0

    @property
    def dim_total(self) -> int:
        return self.dim_h + self.dim_g + self.dim_c
