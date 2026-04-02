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
from slide_cube.util import OUTPUT_ROOT, display_param, load_ground_truth_json, make_model, point_mse_loss, rollout_position_trajectory, rollout_state_trajectory


def parse_args(argv=None):
    argv = resolve_config_argv(argv, script_file=__file__, default_config_name="sysid_cfg.json", config_section="single")
    p = argparse.ArgumentParser(description="Single-run friction sysID for the URDF-backed slide_cube contact system.")
    p.add_argument("--gt-json", type=str, required=True)
    p.add_argument("--fit-param", default="friction_coeff", choices=["friction_coeff"])
    p.add_argument("--init-value", type=float, required=True)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--lr", type=float, default=0.25)
    p.add_argument("--min-value", type=float, default=0.0)
    p.add_argument("--max-value", type=float, default=2.0)
    p.add_argument("--output-json", type=str, default=str(OUTPUT_ROOT / "sysid" / "result.json"))
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
    mu_grad = float(np.mean(model.shape_material_mu.grad.numpy()[:3]))
    out = {
        "loss": float(loss.numpy()[0]),
        "rmse": float(np.sqrt(np.mean((pred_np - target_np) ** 2))),
        "grad": mu_grad,
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


def pack_param_section(value: float, *, include_metrics: dict | None = None):
    payload = {
        "friction_coeff": display_param("friction_coeff", value, "identity"),
        "friction_coeff_raw": float(value),
    }
    if include_metrics is not None:
        payload["loss"] = float(include_metrics["loss"])
        payload["rmse"] = float(include_metrics["rmse"])
        payload["grad"] = float(include_metrics["grad"])
    return payload


def main():
    args = parse_args()
    gt_data = load_ground_truth_json(args.gt_json)
    current = float(args.init_value)
    history = []

    for i in range(args.iters):
        result = evaluate(gt_data, current)
        history.append(
            {
                "iter": i,
                "param_value": float(current),
                "loss": result["loss"],
                "rmse": result["rmse"],
                "grad": result["grad"],
            }
        )
        current = float(np.clip(current - args.lr * result["grad"], args.min_value, args.max_value))

    initial = evaluate(gt_data, float(args.init_value))
    final = evaluate(gt_data, current)
    gt_value = float(gt_data["ground_truth_raw"]["friction_coeff"])
    init_q, init_qd = replay_variant(gt_data, float(args.init_value))
    fit_q, fit_qd = replay_variant(gt_data, current)
    out = {
        "asset": str(Path(gt_data["data"].get("asset") or "")),
        "system": "slide_cube",
        "fit_param": "friction_coeff",
        "fit_params": ["friction_coeff"],
        "angle_convention": gt_data["data"].get("angle_convention"),
        "ground_truth_source": gt_data["path"],
        "ground_truth": pack_param_section(gt_value),
        "fixed_parameters": {k: float(v) for k, v in gt_data["fixed"].items()},
        "initial_guess": pack_param_section(float(args.init_value), include_metrics=initial),
        "final_fit": pack_param_section(current, include_metrics=final),
        "config": {
            "iters": int(args.iters),
            "lr": float(args.lr),
            "steps": gt_data["steps"],
            "dt": gt_data["dt"],
            "solver_iterations": gt_data["solver_iterations"],
            "clamp": [float(args.min_value), float(args.max_value)],
        },
        "metrics": {
            "initial_loss": initial["loss"],
            "final_loss": final["loss"],
            "initial_rmse": initial["rmse"],
            "final_rmse": final["rmse"],
            "final_grad": final["grad"],
            "loss_improvement": initial["loss"] - final["loss"],
            "rmse_improvement": initial["rmse"] - final["rmse"],
        },
        "history": history,
        "target_trajectory": gt_data["target_trajectory"].tolist(),
        "initial_prediction": initial["trajectory"].tolist(),
        "final_prediction": final["trajectory"].tolist(),
        "replay": {
            "time": gt_data["time"].tolist(),
            "variants": {
                "gt": {"body_q": gt_data["gt_q"].tolist(), "body_qd": gt_data["gt_qd"].tolist()},
                "init": {"body_q": init_q.tolist(), "body_qd": init_qd.tolist()},
                "fit": {"body_q": fit_q.tolist(), "body_qd": fit_qd.tolist()},
            },
        },
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    wp.init()
    main()
