from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BinaryPSOConfig:
    n_particles: int
    n_iterations: int
    inertia: float
    c1: float
    c2: float
    min_selected: int
    feature_penalty: float
    seed: int


@dataclass
class BinaryPSOResult:
    best_indices: list[int]
    best_score: float
    history: list[dict[str, float]]


def enforce_minimum_features(position: np.ndarray, min_selected: int) -> np.ndarray:
    if position.size <= min_selected:
        return np.ones_like(position, dtype=bool)
    mask = position >= 0.5
    if mask.sum() >= min_selected:
        return mask
    top_indices = np.argsort(position)[-min_selected:]
    mask[top_indices] = True
    return mask


def run_binary_pso(
    n_dimensions: int,
    config: BinaryPSOConfig,
    objective_fn: Callable[[list[int]], float],
) -> BinaryPSOResult:
    rng = np.random.default_rng(config.seed)
    positions = rng.uniform(0.0, 1.0, size=(config.n_particles, n_dimensions))
    velocities = rng.uniform(-0.2, 0.2, size=(config.n_particles, n_dimensions))

    personal_best_positions = positions.copy()
    personal_best_scores = np.full(config.n_particles, np.inf, dtype=np.float64)

    global_best_position = positions[0].copy()
    global_best_score = float("inf")
    history: list[dict[str, float]] = []

    for iteration in range(config.n_iterations):
        for particle in range(config.n_particles):
            mask = enforce_minimum_features(positions[particle], config.min_selected)
            indices = np.flatnonzero(mask).tolist()
            score = objective_fn(indices) + config.feature_penalty * (len(indices) / n_dimensions)

            if score < personal_best_scores[particle]:
                personal_best_scores[particle] = score
                personal_best_positions[particle] = positions[particle].copy()

            if score < global_best_score:
                global_best_score = score
                global_best_position = positions[particle].copy()

        history.append({"iteration": float(iteration + 1), "best_score": float(global_best_score)})

        r1 = rng.random(size=(config.n_particles, n_dimensions))
        r2 = rng.random(size=(config.n_particles, n_dimensions))
        velocities = (
            config.inertia * velocities
            + config.c1 * r1 * (personal_best_positions - positions)
            + config.c2 * r2 * (global_best_position - positions)
        )
        positions = np.clip(positions + velocities, 0.0, 1.0)

    final_mask = enforce_minimum_features(global_best_position, config.min_selected)
    return BinaryPSOResult(
        best_indices=np.flatnonzero(final_mask).tolist(),
        best_score=float(global_best_score),
        history=history,
    )
