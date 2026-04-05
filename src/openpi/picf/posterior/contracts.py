from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class PointExpertState:
    mu: np.ndarray
    var_block: np.ndarray
    gate: np.ndarray
    anchor_count: np.ndarray
    gamma_n: np.ndarray
    gamma_pc: np.ndarray
    delta_pc: np.ndarray
    delta2x: np.ndarray

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float32)
        self.var_block = np.asarray(self.var_block, dtype=np.float32)
        self.gate = np.asarray(self.gate, dtype=bool)
        self.anchor_count = np.asarray(self.anchor_count, dtype=np.int32)
        self.gamma_n = np.asarray(self.gamma_n, dtype=np.float32)
        self.gamma_pc = np.asarray(self.gamma_pc, dtype=np.float32)
        self.delta_pc = np.asarray(self.delta_pc, dtype=np.float32)
        self.delta2x = np.asarray(self.delta2x, dtype=np.float32)


@dataclasses.dataclass
class PosteriorDebugMetrics:
    point_gate_ratio: float
    stale_prior_match_error: float
    posterior_prior_equal_on_stale: bool
    matched_prior_count: int
    reset_prior_count: int
    precision_gain_count: int
    nan_count: int
    max_abs_mu: float
    min_var_block: float
    max_var_block: float


@dataclasses.dataclass
class PosteriorState:
    mu: np.ndarray
    var_block: np.ndarray
    mu_prop: np.ndarray
    var_prop_block: np.ndarray
    point: PointExpertState
    step_id: int
    segment_id: int
    debug: PosteriorDebugMetrics

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float32)
        self.var_block = np.asarray(self.var_block, dtype=np.float32)
        self.mu_prop = np.asarray(self.mu_prop, dtype=np.float32)
        self.var_prop_block = np.asarray(self.var_prop_block, dtype=np.float32)
