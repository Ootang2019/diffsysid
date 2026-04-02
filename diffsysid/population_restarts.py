from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class SharedRestartEvent:
    iteration: int
    restarted_envs: list[int]
    parent_envs: list[int]
    elite_envs: list[int]
    random_restart_envs: list[int]
    best_env_loss: float
    worst_env_loss: float
    noise_scale: float


@dataclass
class SharedPopulationMetrics:
    best_env_loss: float
    median_env_loss: float
    worst_env_loss: float
    population_std: list[float]
    distance_to_target_mean: list[float]
    distance_to_target_best: list[float]
    within_tol_fraction: float
    best_env_index: int
    worst_env_index: int


TransformFn = Callable[[np.ndarray], np.ndarray]
CloneFn = Callable[[np.ndarray, int, int, float, np.random.Generator], None]
RandomRestartFn = Callable[[np.ndarray, int, np.ndarray, float, np.random.Generator], None]


def compute_population_metrics(params: np.ndarray, target_params: np.ndarray, env_loss: np.ndarray, collapse_tol: float) -> SharedPopulationMetrics:
    params_2d = np.asarray(params, dtype=np.float64)
    if params_2d.ndim == 1:
        params_2d = params_2d[:, None]

    target_2d = np.asarray(target_params, dtype=np.float64)
    if target_2d.ndim == 1:
        if target_2d.shape[0] == params_2d.shape[0]:
            target_2d = target_2d[:, None]
        else:
            target_2d = np.broadcast_to(target_2d[None, :], params_2d.shape)
    elif target_2d.shape[0] == 1 and params_2d.shape[0] != 1:
        target_2d = np.broadcast_to(target_2d, params_2d.shape)

    dist = np.abs(params_2d - target_2d)
    best_env = int(np.argmin(env_loss))
    worst_env = int(np.argmax(env_loss))
    within_tol = np.all(dist <= collapse_tol, axis=1)

    return SharedPopulationMetrics(
        best_env_loss=float(np.min(env_loss)),
        median_env_loss=float(np.median(env_loss)),
        worst_env_loss=float(np.max(env_loss)),
        population_std=np.std(params_2d, axis=0).astype(float).tolist(),
        distance_to_target_mean=np.mean(dist, axis=0).astype(float).tolist(),
        distance_to_target_best=dist[best_env].astype(float).tolist(),
        within_tol_fraction=float(np.mean(within_tol.astype(np.float32))),
        best_env_index=best_env,
        worst_env_index=worst_env,
    )


def annealed_noise_scale(iteration: int, restart_period: int, initial_noise_scale: float, min_noise_scale: float, decay: float) -> float:
    if restart_period <= 0:
        return float(max(min_noise_scale, initial_noise_scale))
    anneal = decay ** (iteration // max(1, restart_period))
    return float(max(min_noise_scale, initial_noise_scale * anneal))


def apply_elite_restarts(raw_params: np.ndarray, moment1: np.ndarray, moment2: np.ndarray, env_loss: np.ndarray, iteration: int, restart_period: int, elite_count: int, restart_count: int, clone_noise_scale: float, rng: np.random.Generator, *, random_restart_fraction: float = 0.0, init_raw_params: np.ndarray | None = None, clip_min: np.ndarray | float | None = None, clip_max: np.ndarray | float | None = None, rank_weighted_elites: bool = False, clone_fn: CloneFn | None = None, random_restart_fn: RandomRestartFn | None = None) -> SharedRestartEvent | None:
    if restart_period <= 0 or restart_count <= 0 or elite_count <= 0:
        return None
    if iteration <= 0 or iteration % restart_period != 0:
        return None

    raw_params = np.asarray(raw_params)
    env_count = raw_params.shape[0]
    elite_count = max(1, min(elite_count, env_count))
    restart_count = max(0, min(restart_count, env_count - elite_count))
    if restart_count == 0:
        return None

    order = np.argsort(env_loss)
    elite_envs = order[:elite_count]
    restarted_envs = order[-restart_count:]

    weights = None
    if rank_weighted_elites:
        weights = np.arange(elite_count, 0, -1, dtype=np.float64)
        weights /= np.sum(weights)

    random_restart_fraction = float(np.clip(random_restart_fraction, 0.0, 1.0))
    random_restart_count = min(int(round(restart_count * random_restart_fraction)), restart_count)
    random_restart_envs = set(rng.choice(restarted_envs, size=random_restart_count, replace=False).tolist()) if random_restart_count > 0 else set()

    parent_envs: list[int] = []
    random_restart_list: list[int] = []
    for env_id_np in restarted_envs:
        env_id = int(env_id_np)
        if env_id in random_restart_envs:
            if init_raw_params is None:
                raise ValueError('init_raw_params is required when random_restart_fraction > 0')
            template_env = int(rng.integers(0, np.asarray(init_raw_params).shape[0]))
            if random_restart_fn is None:
                noise = rng.normal(loc=0.0, scale=clone_noise_scale, size=np.asarray(raw_params[env_id]).shape)
                raw_params[env_id] = np.asarray(init_raw_params)[template_env] + noise
            else:
                random_restart_fn(raw_params, env_id, np.asarray(init_raw_params)[template_env], clone_noise_scale, rng)
            parent_envs.append(-1)
            random_restart_list.append(env_id)
        else:
            parent = int(rng.choice(elite_envs, p=weights)) if weights is not None else int(rng.choice(elite_envs))
            if clone_fn is None:
                noise = rng.normal(loc=0.0, scale=clone_noise_scale, size=np.asarray(raw_params[env_id]).shape)
                raw_params[env_id] = raw_params[parent] + noise
            else:
                clone_fn(raw_params, env_id, parent, clone_noise_scale, rng)
            parent_envs.append(parent)

        if clip_min is not None or clip_max is not None:
            raw_params[env_id] = np.clip(raw_params[env_id], clip_min, clip_max)
        moment1[env_id] = 0.0
        moment2[env_id] = 0.0

    return SharedRestartEvent(
        iteration=iteration,
        restarted_envs=[int(x) for x in restarted_envs.tolist()],
        parent_envs=parent_envs,
        elite_envs=[int(x) for x in elite_envs.tolist()],
        random_restart_envs=random_restart_list,
        best_env_loss=float(env_loss[order[0]]),
        worst_env_loss=float(env_loss[order[-1]]),
        noise_scale=float(clone_noise_scale),
    )
