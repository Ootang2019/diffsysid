from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class TrajectorySnippet:
    snippet_index: int
    start_step: int
    end_step: int
    steps: int
    point_count: int
    fixed: dict[str, float]
    target_traj_np: np.ndarray
    target_traj: wp.array


def resolve_snippet_steps(total_steps: int, dt: float, snippet_steps: int | None, snippet_duration: float | None) -> int:
    if snippet_steps is not None and snippet_duration is not None:
        raise ValueError("Specify at most one of --snippet-steps and --snippet-duration.")
    if total_steps <= 0:
        return 0
    if snippet_steps is not None:
        steps = int(snippet_steps)
    elif snippet_duration is not None and snippet_duration > 0.0:
        steps = int(round(float(snippet_duration) / float(dt)))
    else:
        return total_steps
    if steps <= 0:
        raise ValueError("snippet length must be positive.")
    return min(total_steps, steps)


def build_trajectory_snippets(
    *,
    target_traj_np: np.ndarray,
    gt_q: np.ndarray,
    gt_qd: np.ndarray,
    base_fixed: dict[str, float],
    q_param_names: list[str],
    qd_param_names: list[str],
    snippet_steps: int,
) -> list[TrajectorySnippet]:
    total_steps = int(target_traj_np.shape[0] - 1)
    if total_steps <= 0:
        raise ValueError("target trajectory must contain at least two points.")
    if snippet_steps <= 0:
        raise ValueError("snippet_steps must be positive.")

    snippets: list[TrajectorySnippet] = []
    snippet_index = 0
    for start_step in range(0, total_steps, snippet_steps):
        end_step = min(total_steps, start_step + snippet_steps)
        fixed = dict(base_fixed)
        for i, name in enumerate(q_param_names):
            fixed[name] = float(gt_q[start_step, i])
        for i, name in enumerate(qd_param_names):
            fixed[name] = float(gt_qd[start_step, i])
        snippet_traj_np = np.asarray(target_traj_np[start_step : end_step + 1], dtype=np.float32).copy()
        snippets.append(
            TrajectorySnippet(
                snippet_index=snippet_index,
                start_step=start_step,
                end_step=end_step,
                steps=end_step - start_step,
                point_count=int(snippet_traj_np.shape[0]),
                fixed=fixed,
                target_traj_np=snippet_traj_np,
                target_traj=wp.array(snippet_traj_np, dtype=float),
            )
        )
        snippet_index += 1
    return snippets


def evaluate_population_snippets(
    *,
    snippets: list[TrajectorySnippet],
    fit_params: list[str],
    raw_params: np.ndarray,
    dt: float,
    evaluate_fn,
):
    metrics = []
    losses = np.empty(raw_params.shape[0], dtype=np.float64)
    rmses = np.empty(raw_params.shape[0], dtype=np.float64)
    grads = np.empty((raw_params.shape[0], len(fit_params)), dtype=np.float64)

    for env_index, param_values in enumerate(raw_params):
        total_weight = 0.0
        loss_acc = 0.0
        mse_acc = 0.0
        grad_acc = np.zeros(len(fit_params), dtype=np.float64)
        snippet_metrics = []

        for snippet in snippets:
            metric = evaluate_fn(snippet.target_traj, fit_params, np.asarray(param_values, dtype=np.float64), snippet.fixed, snippet.steps, dt)
            weight = float(snippet.point_count)
            total_weight += weight
            loss_acc += float(metric["loss"]) * weight
            mse_acc += float(metric["rmse"]) ** 2 * weight
            grad_acc += np.asarray([metric["grads"][name] for name in fit_params], dtype=np.float64) * weight
            snippet_metrics.append(
                {
                    "snippet_index": snippet.snippet_index,
                    "start_step": snippet.start_step,
                    "end_step": snippet.end_step,
                    "loss": float(metric["loss"]),
                    "rmse": float(metric["rmse"]),
                }
            )

        mean_loss = loss_acc / total_weight
        mean_grads = grad_acc / total_weight
        env_metric = {
            "loss": float(mean_loss),
            "rmse": float(np.sqrt(mse_acc / total_weight)),
            "grads": {name: float(mean_grads[i]) for i, name in enumerate(fit_params)},
            "snippet_metrics": snippet_metrics,
        }
        if len(fit_params) == 1:
            env_metric["grad"] = env_metric["grads"][fit_params[0]]
        metrics.append(env_metric)
        losses[env_index] = env_metric["loss"]
        rmses[env_index] = env_metric["rmse"]
        grads[env_index] = mean_grads

    return metrics, losses, rmses, grads
