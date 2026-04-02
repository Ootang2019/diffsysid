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

from config_cli import resolve_config_argv
from diffsysid.population_restarts import annealed_noise_scale, apply_elite_restarts, compute_population_metrics
from slide_cube.util import OUTPUT_ROOT, display_param, load_ground_truth_json, make_model, point_mse_loss, rollout_position_trajectory, rollout_state_trajectory


def parse_args(argv=None):
    argv = resolve_config_argv(argv, script_file=__file__, default_config_name="sysid_cfg.json", config_section="batch")
    p = argparse.ArgumentParser(description="Population batch friction sysID for the URDF-backed slide_cube contact system.")
    p.add_argument("--gt-json", type=str, required=True)
    p.add_argument("--fit-param", default="friction_coeff", choices=["friction_coeff"])
    p.add_argument("--init-value", type=float, required=True)
    p.add_argument("--init-span", type=float, default=0.2)
    p.add_argument("--env-count", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1e4)
    p.add_argument("--restart-period", type=int, default=10)
    p.add_argument("--elite-count", type=int, default=2)
    p.add_argument("--restart-count", type=int, default=3)
    p.add_argument("--random-restart-fraction", type=float, default=0.34)
    p.add_argument("--clone-noise-scale", type=float, default=0.05)
    p.add_argument("--min-clone-noise-scale", type=float, default=0.005)
    p.add_argument("--clone-decay", type=float, default=0.8)
    p.add_argument("--collapse-tol", type=float, default=0.02)
    p.add_argument("--clip-min", type=float, default=0.0)
    p.add_argument("--clip-max", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, default=str(OUTPUT_ROOT / "batch_sysid" / "result.json"))
    return p.parse_args(argv)


def evaluate(gt_data: dict, friction_coeff: float):
    target_np = gt_data["target_trajectory"]
    target = wp.array(target_np, dtype=float)
    model = make_model(friction_coeff=friction_coeff, requires_grad=True, **gt_data["fixed"])
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        pred = rollout_position_trajectory(
            model,
            steps=gt_data["steps"],
            dt=gt_data["dt"],
            solver_iterations=gt_data["solver_iterations"],
            requires_grad=True,
        )
        wp.launch(point_mse_loss, dim=gt_data["steps"] + 1, inputs=[pred, target, gt_data["steps"] + 1, loss])
    tape.backward(loss)
    pred_np = pred.numpy()
    grad = float(np.mean(model.shape_material_mu.grad.numpy()[:3]))
    out = {
        "loss": float(loss.numpy()[0]),
        "rmse": float(np.sqrt(np.mean((pred_np - target_np) ** 2))),
        "grad": grad,
        "trajectory": pred_np,
        "final_position": pred_np[-1].tolist(),
    }
    tape.zero()
    return out


def replay_variant(gt_data: dict, friction_coeff: float):
    model = make_model(friction_coeff=friction_coeff, requires_grad=False, **gt_data["fixed"])
    _, q, qd = rollout_state_trajectory(
        model,
        steps=gt_data["steps"],
        dt=gt_data["dt"],
        solver_iterations=gt_data["solver_iterations"],
        requires_grad=False,
    )
    return q, qd


def pack_param_section(value: float, *, include_metrics: dict | None = None, env_index: int | None = None):
    payload = {
        "friction_coeff": display_param("friction_coeff", value, "identity"),
        "friction_coeff_raw": float(value),
    }
    if include_metrics is not None:
        payload["loss"] = float(include_metrics["loss"])
        payload["rmse"] = float(include_metrics["rmse"])
        payload["grad"] = float(include_metrics["grad"])
    if env_index is not None:
        payload["env_index"] = int(env_index)
    return payload


def main():
    args = parse_args()
    gt_data = load_ground_truth_json(args.gt_json)
    rng = np.random.default_rng(args.seed)
    raw_params = float(args.init_value) + rng.uniform(-args.init_span, args.init_span, size=(args.env_count, 1))
    raw_params[0, 0] = float(args.init_value)
    raw_params = np.clip(raw_params, args.clip_min, args.clip_max)
    init_raw_params = raw_params.copy()
    moment1 = np.zeros_like(raw_params)
    moment2 = np.zeros_like(raw_params)
    target_params = np.full_like(raw_params, float(gt_data["ground_truth_raw"]["friction_coeff"]))

    initial_metrics = [evaluate(gt_data, float(v[0])) for v in raw_params]
    initial_losses = np.array([m["loss"] for m in initial_metrics], dtype=np.float64)
    initial_rmses = np.array([m["rmse"] for m in initial_metrics], dtype=np.float64)
    initial_best_idx = int(np.argmin(initial_losses))
    best = {
        "param_value": float(raw_params[initial_best_idx, 0]),
        "metrics": initial_metrics[initial_best_idx],
        "env_index": initial_best_idx,
        "iteration": -1,
    }

    history = []
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for it in range(args.iters):
        metrics = [evaluate(gt_data, float(v[0])) for v in raw_params]
        env_loss = np.array([m["loss"] for m in metrics], dtype=np.float64)
        env_rmse = np.array([m["rmse"] for m in metrics], dtype=np.float64)
        env_grad = np.array([[m["grad"]] for m in metrics], dtype=np.float64)
        best_idx = int(np.argmin(env_loss))
        if env_loss[best_idx] < best["metrics"]["loss"]:
            best = {"param_value": float(raw_params[best_idx, 0]), "metrics": metrics[best_idx], "env_index": best_idx, "iteration": it}

        pop = compute_population_metrics(raw_params, target_params, env_loss, args.collapse_tol)
        history.append(
            {
                "iteration": it,
                "best_env_index": best_idx,
                "best_loss": float(env_loss[best_idx]),
                "mean_loss": float(np.mean(env_loss)),
                "best_rmse": float(env_rmse[best_idx]),
                "best_param_value": float(raw_params[best_idx, 0]),
                "median_loss": pop.median_env_loss,
                "worst_loss": pop.worst_env_loss,
                "population_std": pop.population_std,
                "distance_to_target_mean": pop.distance_to_target_mean,
                "distance_to_target_best": pop.distance_to_target_best,
                "within_tol_fraction": pop.within_tol_fraction,
            }
        )
        print(f"iter={it:03d} best_loss={env_loss[best_idx]:.6e} mean_loss={np.mean(env_loss):.6e} best_rmse={env_rmse[best_idx]:.6e} friction_coeff={raw_params[best_idx,0]:.6f}")

        grad = np.clip(env_grad, -args.grad_clip, args.grad_clip)
        moment1 = beta1 * moment1 + (1.0 - beta1) * grad
        moment2 = beta2 * moment2 + (1.0 - beta2) * (grad * grad)
        m_hat = moment1 / (1.0 - beta1 ** (it + 1))
        v_hat = moment2 / (1.0 - beta2 ** (it + 1))
        raw_params -= args.lr * m_hat / (np.sqrt(v_hat) + eps)
        raw_params = np.clip(raw_params, args.clip_min, args.clip_max)

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
            history[-1]["restart_event"] = {
                "restarted_envs": restart.restarted_envs,
                "parent_envs": restart.parent_envs,
                "elite_envs": restart.elite_envs,
                "random_restart_envs": restart.random_restart_envs,
                "noise_scale": restart.noise_scale,
            }

    final_metrics_all = [evaluate(gt_data, float(v[0])) for v in raw_params]
    final_losses = np.array([m["loss"] for m in final_metrics_all], dtype=np.float64)
    final_rmses = np.array([m["rmse"] for m in final_metrics_all], dtype=np.float64)
    final_best_idx = int(np.argmin(final_losses))
    if final_losses[final_best_idx] < best["metrics"]["loss"]:
        best = {"param_value": float(raw_params[final_best_idx, 0]), "metrics": final_metrics_all[final_best_idx], "env_index": final_best_idx, "iteration": args.iters}

    init_param_value = float(init_raw_params[initial_best_idx, 0])
    fit_param_value = float(best["param_value"])
    init_q, init_qd = replay_variant(gt_data, init_param_value)
    fit_q, fit_qd = replay_variant(gt_data, fit_param_value)
    gt_value = float(gt_data["ground_truth_raw"]["friction_coeff"])

    result = {
        "asset": str(Path(gt_data["data"].get("asset") or "")),
        "system": "slide_cube",
        "fit_param": "friction_coeff",
        "fit_params": ["friction_coeff"],
        "angle_convention": gt_data["data"].get("angle_convention"),
        "ground_truth_source": gt_data["path"],
        "ground_truth": pack_param_section(gt_value),
        "fixed_parameters": {k: float(v) for k, v in gt_data["fixed"].items()},
        "initial_guess": pack_param_section(init_param_value, include_metrics=initial_metrics[initial_best_idx], env_index=initial_best_idx),
        "initial_population": {
            "center_raw": [float(args.init_value)],
            "span": [float(args.init_span)],
            "env_count": int(args.env_count),
            "raw_params": init_raw_params.astype(float).tolist(),
        },
        "final_fit": {**pack_param_section(fit_param_value, include_metrics=best["metrics"]), "best_env_index": best["env_index"], "best_iteration": best["iteration"]},
        "final_population": {
            "raw_params": raw_params.astype(float).tolist(),
            "best_env_index": final_best_idx,
            "best_loss": float(final_losses[final_best_idx]),
            "best_rmse": float(final_rmses[final_best_idx]),
        },
        "config": vars(args),
        "history": history,
        "target_trajectory": gt_data["target_trajectory"].tolist(),
        "initial_prediction": initial_metrics[initial_best_idx]["trajectory"].tolist(),
        "final_prediction": best["metrics"]["trajectory"].tolist(),
        "replay": {
            "time": gt_data["time"].tolist(),
            "variants": {
                "gt": {"body_q": gt_data["gt_q"].tolist(), "body_qd": gt_data["gt_qd"].tolist()},
                "init": {"body_q": init_q.tolist(), "body_qd": init_qd.tolist()},
                "fit": {"body_q": fit_q.tolist(), "body_qd": fit_qd.tolist()},
            },
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved results to {out}")


if __name__ == "__main__":
    wp.init()
    main()
