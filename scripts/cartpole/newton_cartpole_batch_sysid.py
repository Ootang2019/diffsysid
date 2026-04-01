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
from newton_cartpole_sysid import (
    evaluate,
    make_model_with_param,
    rollout_joint_state_trajectory,
    rollout_tip_trajectory,
    top_offset_to_raw_angle,
    raw_angle_to_top_offset,
)
from util import URDF_PATH


def display_value(raw_value: float, *, fit_param: str, angle_mode: str) -> float:
    if angle_mode == "top_offset" and fit_param == "init_pole_angle":
        return raw_angle_to_top_offset(raw_value)
    return raw_value


def evaluate_population(target_traj, fit_param: str, raw_params: np.ndarray, fixed: dict[str, float], steps: int, dt: float):
    metrics = [evaluate(target_traj, fit_param, float(param), fixed, steps, dt) for param in raw_params]
    losses = np.asarray([m["loss"] for m in metrics], dtype=np.float64)
    rmses = np.asarray([m["rmse"] for m in metrics], dtype=np.float64)
    grads = np.asarray([m["grad"] for m in metrics], dtype=np.float64)
    return metrics, losses, rmses, grads


def run(args):
    gt_value_raw = top_offset_to_raw_angle(args.gt_value) if args.angle_mode == "top_offset" and args.fit_param == "init_pole_angle" else args.gt_value
    init_value_raw = top_offset_to_raw_angle(args.init_value) if args.angle_mode == "top_offset" and args.fit_param == "init_pole_angle" else args.init_value
    init_pole_angle_raw = top_offset_to_raw_angle(args.init_pole_angle) if args.angle_mode == "top_offset" else args.init_pole_angle

    fixed = {
        "init_cart_pos": args.init_cart_pos,
        "init_pole_angle": gt_value_raw if args.fit_param == "init_pole_angle" else init_pole_angle_raw,
        "init_cart_vel": args.init_cart_vel,
        "init_pole_angvel": args.init_pole_angvel,
    }

    gt_model, _, _ = make_model_with_param(args.fit_param, gt_value_raw, fixed, requires_grad=False)
    gt_traj_np = rollout_tip_trajectory(gt_model, steps=args.steps, dt=args.dt, requires_grad=False).numpy()
    gt_traj = wp.array(gt_traj_np, dtype=float)

    rng = np.random.default_rng(args.seed)
    raw_params = init_value_raw + rng.uniform(-args.init_span, args.init_span, size=args.env_count)
    raw_params[0] = init_value_raw
    init_raw_params = raw_params.copy()
    moment1 = np.zeros_like(raw_params)
    moment2 = np.zeros_like(raw_params)

    history: list[dict] = []
    initial_metrics, initial_losses, initial_rmses, _ = evaluate_population(gt_traj, args.fit_param, raw_params, fixed, args.steps, args.dt)
    initial_best_idx = int(np.argmin(initial_losses))
    best = {
        "param_value": float(raw_params[initial_best_idx]),
        "metrics": initial_metrics[initial_best_idx],
        "env_index": initial_best_idx,
        "iteration": -1,
    }

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    target_params = np.full(args.env_count, gt_value_raw, dtype=np.float64)

    for it in range(args.iters):
        metrics, env_loss, env_rmse, env_grad = evaluate_population(gt_traj, args.fit_param, raw_params, fixed, args.steps, args.dt)
        best_idx = int(np.argmin(env_loss))
        if env_loss[best_idx] < best["metrics"]["loss"]:
            best = {"param_value": float(raw_params[best_idx]), "metrics": metrics[best_idx], "env_index": best_idx, "iteration": it}

        pop = compute_population_metrics(raw_params, target_params, env_loss, args.collapse_tol)
        history.append(
            {
                "iteration": it,
                "best_env_index": best_idx,
                "best_loss": float(env_loss[best_idx]),
                "best_rmse": float(env_rmse[best_idx]),
                "best_param_value": float(raw_params[best_idx]),
                "median_loss": pop.median_env_loss,
                "worst_loss": pop.worst_env_loss,
                "population_std": pop.population_std,
                "distance_to_target_mean": pop.distance_to_target_mean,
                "distance_to_target_best": pop.distance_to_target_best,
                "within_tol_fraction": pop.within_tol_fraction,
            }
        )
        print(
            f"iter={it:03d} best_loss={env_loss[best_idx]:.6e} median_loss={pop.median_env_loss:.6e} "
            f"best_rmse={env_rmse[best_idx]:.6e} best_{args.fit_param}={raw_params[best_idx]:.6f}"
        )

        grad = np.clip(env_grad, -args.grad_clip, args.grad_clip)
        moment1 = beta1 * moment1 + (1.0 - beta1) * grad
        moment2 = beta2 * moment2 + (1.0 - beta2) * (grad * grad)
        m_hat = moment1 / (1.0 - beta1 ** (it + 1))
        v_hat = moment2 / (1.0 - beta2 ** (it + 1))
        raw_params -= args.lr * m_hat / (np.sqrt(v_hat) + eps)

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

    final_metrics_all, final_losses, final_rmses, _ = evaluate_population(gt_traj, args.fit_param, raw_params, fixed, args.steps, args.dt)
    final_best_idx = int(np.argmin(final_losses))
    if final_losses[final_best_idx] < best["metrics"]["loss"]:
        best = {"param_value": float(raw_params[final_best_idx]), "metrics": final_metrics_all[final_best_idx], "env_index": final_best_idx, "iteration": args.iters}

    init_param = float(init_raw_params[initial_best_idx])
    fit_param = float(best["param_value"])

    gt_replay_model, _, _ = make_model_with_param(args.fit_param, gt_value_raw, fixed, requires_grad=False)
    init_replay_model, _, _ = make_model_with_param(args.fit_param, init_param, fixed, requires_grad=False)
    fit_replay_model, _, _ = make_model_with_param(args.fit_param, fit_param, fixed, requires_grad=False)
    gt_q, gt_qd = rollout_joint_state_trajectory(gt_replay_model, args.steps, args.dt)
    init_q, init_qd = rollout_joint_state_trajectory(init_replay_model, args.steps, args.dt)
    fit_q, fit_qd = rollout_joint_state_trajectory(fit_replay_model, args.steps, args.dt)

    result = {
        "asset": str(URDF_PATH),
        "fit_param": args.fit_param,
        "angle_convention": {
            "mode": args.angle_mode,
            "raw_zero_meaning": "pole hanging straight down",
            "raw_upright_angle": float(np.pi),
        },
        "ground_truth": {args.fit_param: display_value(gt_value_raw, fit_param=args.fit_param, angle_mode=args.angle_mode), f"{args.fit_param}_raw": gt_value_raw},
        "initial_guess": {
            args.fit_param: display_value(init_param, fit_param=args.fit_param, angle_mode=args.angle_mode),
            f"{args.fit_param}_raw": init_param,
            "loss": float(initial_losses[initial_best_idx]),
            "rmse": float(initial_rmses[initial_best_idx]),
            "env_index": initial_best_idx,
        },
        "initial_population": {
            "center_raw": init_value_raw,
            "span": args.init_span,
            "env_count": args.env_count,
            "raw_params": init_raw_params.astype(float).tolist(),
        },
        "final_fit": {
            args.fit_param: display_value(fit_param, fit_param=args.fit_param, angle_mode=args.angle_mode),
            f"{args.fit_param}_raw": fit_param,
            "loss": float(best["metrics"]["loss"]),
            "rmse": float(best["metrics"]["rmse"]),
            "grad": float(best["metrics"]["grad"]),
            "best_env_index": best["env_index"],
            "best_iteration": best["iteration"],
        },
        "final_population": {
            "raw_params": raw_params.astype(float).tolist(),
            "best_env_index": final_best_idx,
            "best_loss": float(final_losses[final_best_idx]),
            "best_rmse": float(final_rmses[final_best_idx]),
        },
        "config": vars(args),
        "history": history,
        "target_trajectory": gt_traj_np.tolist(),
        "initial_prediction": initial_metrics[initial_best_idx]["trajectory"].tolist(),
        "final_prediction": best["metrics"]["trajectory"].tolist(),
        "replay": {
            "time": (np.arange(args.steps + 1, dtype=np.float32) * args.dt).tolist(),
            "variants": {
                "gt": {"joint_q": gt_q.tolist(), "joint_qd": gt_qd.tolist()},
                "init": {"joint_q": init_q.tolist(), "joint_qd": init_qd.tolist()},
                "fit": {"joint_q": fit_q.tolist(), "joint_qd": fit_qd.tolist()},
            },
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved results to {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Shared-population batch sysID for the URDF cartpole.")
    p.add_argument("--fit-param", choices=["init_pole_angle", "init_cart_pos", "init_cart_vel", "init_pole_angvel"], required=True)
    p.add_argument("--gt-value", type=float, required=True)
    p.add_argument("--init-value", type=float, required=True)
    p.add_argument("--init-span", type=float, default=0.25)
    p.add_argument("--init-cart-pos", type=float, default=0.0)
    p.add_argument("--init-pole-angle", type=float, default=0.2)
    p.add_argument("--angle-mode", choices=["urdf_raw", "top_offset"], default="urdf_raw")
    p.add_argument("--init-cart-vel", type=float, default=0.0)
    p.add_argument("--init-pole-angvel", type=float, default=0.0)
    p.add_argument("--env-count", type=int, default=8)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--dt", type=float, default=1.0 / 240.0)
    p.add_argument("--iters", type=int, default=40)
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
