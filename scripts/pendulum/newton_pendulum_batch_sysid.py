#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from batch_sysid_common import annealed_noise_scale, apply_elite_restarts, compute_population_metrics
from newton_pendulum_sysid import (
    FIT_PARAM_SELECTION_CHOICES,
    load_or_generate_ground_truth,
    make_model_with_params,
    parameter_to_raw,
    resolve_fit_request,
    rollout_joint_state_trajectory,
    rollout_tip_trajectory,
    display_param,
)
from snippet_batch_sysid_common import build_trajectory_snippets, resolve_snippet_steps
from util import DYNAMIC_PARAM_NAMES, NONNEGATIVE_PARAM_NAMES, PARAM_SPECS, TIP_BODY_INDEX, TIP_LOCAL, URDF_PATH
from world_batch_sysid_common import evaluate_world_batch


def clamp_nonnegative_columns(raw_params: np.ndarray, fit_params: list[str]) -> None:
    for i, name in enumerate(fit_params):
        if name in NONNEGATIVE_PARAM_NAMES:
            raw_params[:, i] = np.maximum(raw_params[:, i], 0.0)


def evaluate_population_batched(
    *,
    fit_params: list[str],
    raw_params: np.ndarray,
    snippets,
    dt: float,
):
    env_count = int(raw_params.shape[0])
    snippet_count = len(snippets)
    max_points = max(snippet.point_count for snippet in snippets)
    world_count = env_count * snippet_count

    target_batch_np = np.zeros((world_count, max_points, 3), dtype=np.float32)
    point_counts = np.zeros(world_count, dtype=np.int32)
    world_to_env = np.zeros(world_count, dtype=np.int32)
    world_fixed = []
    world_snippet_index = np.zeros(world_count, dtype=np.int32)

    world_index = 0
    for snippet in snippets:
        for env_index in range(env_count):
            target_batch_np[world_index, : snippet.point_count] = snippet.target_traj_np
            point_counts[world_index] = snippet.point_count
            world_to_env[world_index] = env_index
            world_fixed.append(dict(snippet.fixed))
            world_snippet_index[world_index] = snippet.snippet_index
            world_index += 1

    batch = evaluate_world_batch(
        urdf_path=URDF_PATH,
        param_specs=PARAM_SPECS,
        fit_params=fit_params,
        per_env_param_values=raw_params,
        world_to_env=world_to_env,
        world_fixed_params=world_fixed,
        target_batch_np=target_batch_np,
        point_counts=point_counts,
        dt=dt,
        tip_body_index=TIP_BODY_INDEX,
        tip_local=TIP_LOCAL,
    )

    metrics = []
    for env_index in range(env_count):
        snippet_metrics = []
        for world_idx in np.nonzero(world_to_env == env_index)[0]:
            snippet = snippets[int(world_snippet_index[world_idx])]
            snippet_metrics.append(
                {
                    "snippet_index": snippet.snippet_index,
                    "start_step": snippet.start_step,
                    "end_step": snippet.end_step,
                    "loss": float(batch.world_losses[world_idx]),
                    "rmse": float(batch.world_rmses[world_idx]),
                }
            )
        env_metric = {
            "loss": float(batch.losses[env_index]),
            "rmse": float(batch.rmses[env_index]),
            "grads": {name: float(batch.grads[env_index, i]) for i, name in enumerate(fit_params)},
        }
        if len(fit_params) == 1:
            env_metric["grad"] = env_metric["grads"][fit_params[0]]
        if len(snippets) > 1:
            env_metric["snippet_metrics"] = snippet_metrics
        else:
            env_metric["trajectory"] = batch.world_trajectories[env_index, : snippets[0].point_count].tolist()
        metrics.append(env_metric)
    return metrics, batch.losses, batch.rmses, batch.grads


def run(args):
    gt_source = load_or_generate_ground_truth(args)
    gt_value_map = dict(zip(gt_source["fit_params"], gt_source["gt_values_raw"]))
    target_value_map = {**gt_source["fixed"], **gt_value_map}
    fit_params, init_values_in = resolve_fit_request(args, default_fit_params=gt_source["fit_params"], target_value_map=target_value_map)
    init_values_raw = np.asarray([parameter_to_raw(name, value, args.angle_mode) for name, value in zip(fit_params, init_values_in)], dtype=np.float64)
    missing = [name for name in fit_params if name not in target_value_map]
    if missing:
        raise ValueError(f"--gt-json is missing ground-truth values for fit params: {missing}")
    fixed = dict(gt_source["fixed"])
    steps = gt_source["steps"]
    dt = gt_source["dt"]
    gt_values_raw = np.asarray([target_value_map[name] for name in fit_params], dtype=np.float64)
    gt_traj_np = gt_source["gt_traj_np"]
    gt_q = gt_source["gt_q"]
    gt_qd = gt_source["gt_qd"]

    snippet_steps = resolve_snippet_steps(steps, dt, args.snippet_steps, args.snippet_duration)
    snippet_count = int(np.ceil(steps / max(1, snippet_steps)))
    if snippet_count > 1 and any(name not in DYNAMIC_PARAM_NAMES for name in fit_params):
        raise ValueError("Long-horizon snippet batching currently supports dynamics parameters only. Initial-state fit params must use the standard full-trajectory batch mode.")
    snippets = build_trajectory_snippets(
        target_traj_np=gt_traj_np,
        gt_q=gt_q,
        gt_qd=gt_qd,
        base_fixed=fixed,
        q_param_names=["init_angle"],
        qd_param_names=["init_angvel"],
        snippet_steps=snippet_steps,
    )

    rng = np.random.default_rng(args.seed)
    init_span = np.asarray(args.init_span, dtype=np.float64)
    if init_span.ndim == 0:
        init_span = np.full(len(fit_params), float(init_span))
    raw_params = init_values_raw[None, :] + rng.uniform(-init_span, init_span, size=(args.env_count, len(fit_params)))
    raw_params[0] = init_values_raw
    clamp_nonnegative_columns(raw_params, fit_params)
    init_raw_params = raw_params.copy()
    moment1 = np.zeros_like(raw_params)
    moment2 = np.zeros_like(raw_params)

    history: list[dict] = []
    initial_metrics, initial_losses, initial_rmses, _ = evaluate_population_batched(
        fit_params=fit_params,
        raw_params=raw_params,
        snippets=snippets,
        dt=dt,
    )
    initial_best_idx = int(np.argmin(initial_losses))
    best = {"param_values": raw_params[initial_best_idx].copy(), "metrics": initial_metrics[initial_best_idx], "env_index": initial_best_idx, "iteration": -1}

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    target_params = np.broadcast_to(gt_values_raw[None, :], raw_params.shape)

    for it in range(args.iters):
        metrics, env_loss, env_rmse, env_grad = evaluate_population_batched(
            fit_params=fit_params,
            raw_params=raw_params,
            snippets=snippets,
            dt=dt,
        )
        best_idx = int(np.argmin(env_loss))
        if env_loss[best_idx] < best["metrics"]["loss"]:
            best = {"param_values": raw_params[best_idx].copy(), "metrics": metrics[best_idx], "env_index": best_idx, "iteration": it}
        pop = compute_population_metrics(raw_params, target_params, env_loss, args.collapse_tol)
        best_params = {name: float(raw_params[best_idx, i]) for i, name in enumerate(fit_params)}
        history.append(
            {
                "iteration": it,
                "best_env_index": best_idx,
                "best_loss": float(env_loss[best_idx]),
                "best_rmse": float(env_rmse[best_idx]),
                "best_param_values": best_params,
                "median_loss": pop.median_env_loss,
                "worst_loss": pop.worst_env_loss,
                "population_std": pop.population_std,
                "distance_to_target_mean": pop.distance_to_target_mean,
                "distance_to_target_best": pop.distance_to_target_best,
                "within_tol_fraction": pop.within_tol_fraction,
                "snippet_count": snippet_count,
                "total_instances": int(args.env_count * snippet_count),
            }
        )
        if len(fit_params) == 1:
            history[-1]["best_param_value"] = float(raw_params[best_idx, 0])
        if "snippet_metrics" in metrics[best_idx]:
            history[-1]["best_snippet_metrics"] = metrics[best_idx]["snippet_metrics"]
        best_status = " ".join(f"{name}={raw_params[best_idx, i]:.6f}" for i, name in enumerate(fit_params))
        print(f"iter={it:03d} best_loss={env_loss[best_idx]:.6e} median_loss={pop.median_env_loss:.6e} best_rmse={env_rmse[best_idx]:.6e} {best_status}")

        grad = np.clip(env_grad, -args.grad_clip, args.grad_clip)
        moment1 = beta1 * moment1 + (1.0 - beta1) * grad
        moment2 = beta2 * moment2 + (1.0 - beta2) * (grad * grad)
        m_hat = moment1 / (1.0 - beta1 ** (it + 1))
        v_hat = moment2 / (1.0 - beta2 ** (it + 1))
        raw_params -= args.lr * m_hat / (np.sqrt(v_hat) + eps)
        clamp_nonnegative_columns(raw_params, fit_params)

        noise_scale = annealed_noise_scale(it + 1, args.restart_period, args.clone_noise_scale, args.min_clone_noise_scale, args.clone_decay)
        restart = apply_elite_restarts(
            raw_params,
            moment1,
            moment2,
            env_loss,
            it + 1,
            args.restart_period,
            args.elite_count,
            args.restart_count,
            noise_scale,
            rng,
            random_restart_fraction=args.random_restart_fraction,
            init_raw_params=init_raw_params,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )
        if restart is not None:
            history[-1]["restart_event"] = {"restarted_envs": restart.restarted_envs, "parent_envs": restart.parent_envs, "elite_envs": restart.elite_envs, "random_restart_envs": restart.random_restart_envs, "noise_scale": restart.noise_scale}

    final_metrics_all, final_losses, final_rmses, _ = evaluate_population_batched(
        fit_params=fit_params,
        raw_params=raw_params,
        snippets=snippets,
        dt=dt,
    )
    final_best_idx = int(np.argmin(final_losses))
    if final_losses[final_best_idx] < best["metrics"]["loss"]:
        best = {"param_values": raw_params[final_best_idx].copy(), "metrics": final_metrics_all[final_best_idx], "env_index": final_best_idx, "iteration": args.iters}

    init_param_values = init_raw_params[initial_best_idx].copy()
    fit_param_values = best["param_values"].copy()

    init_replay_model, _ = make_model_with_params(fit_params, init_param_values, fixed, requires_grad=False)
    fit_replay_model, _ = make_model_with_params(fit_params, fit_param_values, fixed, requires_grad=False)
    init_q, init_qd = rollout_joint_state_trajectory(init_replay_model, steps, dt)
    fit_q, fit_qd = rollout_joint_state_trajectory(fit_replay_model, steps, dt)
    init_traj_np = rollout_tip_trajectory(init_replay_model, steps=steps, dt=dt, requires_grad=False).numpy()
    fit_traj_np = rollout_tip_trajectory(fit_replay_model, steps=steps, dt=dt, requires_grad=False).numpy()

    def pack_param_section(raw_values: np.ndarray, *, include_metrics: dict | None = None, env_index: int | None = None):
        payload = {}
        for i, name in enumerate(fit_params):
            payload[name] = display_param(name, raw_values[i], args.angle_mode)
            payload[f"{name}_raw"] = float(raw_values[i])
        if include_metrics is not None:
            payload["loss"] = float(include_metrics["loss"])
            payload["rmse"] = float(include_metrics["rmse"])
            if len(fit_params) == 1:
                payload["grad"] = float(include_metrics["grads"][fit_params[0]])
            else:
                payload["grads"] = include_metrics["grads"]
            if "snippet_metrics" in include_metrics:
                payload["snippet_metrics"] = include_metrics["snippet_metrics"]
        if env_index is not None:
            payload["env_index"] = int(env_index)
        return payload

    result = {
        "asset": str(URDF_PATH),
        "fit_param": fit_params[0] if len(fit_params) == 1 else None,
        "fit_params": fit_params,
        "angle_convention": {"mode": args.angle_mode, "raw_zero_meaning": "pendulum hanging straight down", "raw_upright_angle": float(np.pi)},
        "ground_truth": pack_param_section(gt_values_raw),
        "initial_guess": pack_param_section(init_param_values, include_metrics=initial_metrics[initial_best_idx], env_index=initial_best_idx),
        "initial_population": {"center_raw": init_values_raw.astype(float).tolist(), "span": init_span.astype(float).tolist(), "env_count": args.env_count, "env_count_per_snippet": args.env_count, "total_instances": int(args.env_count * snippet_count), "raw_params": init_raw_params.astype(float).tolist()},
        "final_fit": {**pack_param_section(fit_param_values, include_metrics=best["metrics"]), "best_env_index": best["env_index"], "best_iteration": best["iteration"]},
        "final_population": {"raw_params": raw_params.astype(float).tolist(), "best_env_index": final_best_idx, "best_loss": float(final_losses[final_best_idx]), "best_rmse": float(final_rmses[final_best_idx])},
        "config": vars(args),
        "ground_truth_source": args.gt_json,
        "snippet_batching": {
            "enabled": bool(snippet_count > 1),
            "snippet_steps": int(snippet_steps),
            "snippet_duration": float(snippet_steps * dt),
            "snippet_count": int(snippet_count),
            "env_count_per_snippet": int(args.env_count),
            "total_instances": int(args.env_count * snippet_count),
            "snippets": [{"snippet_index": snippet.snippet_index, "start_step": snippet.start_step, "end_step": snippet.end_step, "start_time": float(snippet.start_step * dt), "end_time": float(snippet.end_step * dt), "point_count": snippet.point_count} for snippet in snippets],
        },
        "history": history,
        "target_trajectory": gt_traj_np.tolist(),
        "initial_prediction": init_traj_np.tolist(),
        "final_prediction": fit_traj_np.tolist(),
        "replay": {"time": (np.arange(steps + 1, dtype=np.float32) * dt).tolist(), "variants": {"gt": {"joint_q": gt_q.tolist(), "joint_qd": gt_qd.tolist()}, "init": {"joint_q": init_q.tolist(), "joint_qd": init_qd.tolist()}, "fit": {"joint_q": fit_q.tolist(), "joint_qd": fit_qd.tolist()}}},
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved results to {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Shared-population batch sysID for the URDF pendulum.")
    p.add_argument("--fit-param", choices=FIT_PARAM_SELECTION_CHOICES)
    p.add_argument("--init-value", type=str)
    p.add_argument("--fit-params", nargs="+")
    p.add_argument("--init-values", nargs="+")
    p.add_argument("--init-span", type=float, default=0.25)
    p.add_argument("--init-angle", type=float, default=0.2)
    p.add_argument("--angle-mode", choices=["urdf_raw", "top_offset"], default="urdf_raw")
    p.add_argument("--init-angvel", type=float, default=0.0)
    p.add_argument("--hinge-armature", type=float, default=0.0)
    p.add_argument("--hinge-stiffness", type=float, default=0.0)
    p.add_argument("--hinge-damping", type=float, default=0.0)
    p.add_argument("--env-count", type=int, default=8)
    p.add_argument("--gt-json", type=str, required=True)
    p.add_argument("--random-init-span", type=float, default=0.25)
    p.add_argument("--snippet-steps", type=int, default=None)
    p.add_argument("--snippet-duration", type=float, default=None)
    p.add_argument("--iters", "--iter", dest="iters", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1e4)
    p.add_argument("--restart-period", type=int, default=10)
    p.add_argument("--elite-count", type=int, default=2)
    p.add_argument("--restart-count", type=int, default=3)
    p.add_argument("--random-restart-fraction", type=float, default=0.34)
    p.add_argument("--clone-noise-scale", type=float, default=0.05)
    p.add_argument("--min-clone-noise-scale", type=float, default=0.005)
    p.add_argument("--clone-decay", type=float, default=0.8)
    p.add_argument("--collapse-tol", type=float, default=0.02)
    p.add_argument("--clip-min", type=float, default=None)
    p.add_argument("--clip-max", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    wp.init()
    run(parse_args())
