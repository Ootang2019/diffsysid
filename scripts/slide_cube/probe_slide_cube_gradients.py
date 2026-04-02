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
from slide_cube.util import DEFAULTS, OUTPUT_ROOT, load_ground_truth_json, make_model, point_mse_loss, rollout_position_trajectory


def parse_args(argv=None):
    argv = resolve_config_argv(argv, script_file=__file__, default_config_name="sysid_cfg.json", config_section="single")
    p = argparse.ArgumentParser(description="Probe whether slide-cube friction has usable gradients.")
    p.add_argument("--gt-json", type=str, required=True)
    p.add_argument("--fit-param", default="friction_coeff", choices=["friction_coeff"])
    p.add_argument("--init-value", type=float, default=0.6)
    p.add_argument("--probe-values", nargs="+", type=float)
    p.add_argument("--output-json", type=str, default=str(OUTPUT_ROOT / "gradient_probe.json"))
    return p.parse_args(argv)


def evaluate_probe(gt_data: dict, init_value: float):
    target_np = gt_data["target_trajectory"]
    target = wp.array(target_np, dtype=float)
    fixed = dict(gt_data["fixed"])
    model = make_model(friction_coeff=init_value, requires_grad=True, **fixed)
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
    mu_grad = model.shape_material_mu.grad.numpy()
    result = {
        "param": "friction_coeff",
        "gt_value": float(gt_data["ground_truth_raw"]["friction_coeff"]),
        "init_value": float(init_value),
        "loss": float(loss.numpy()[0]),
        "rmse": float(np.sqrt(np.mean((pred_np - target_np) ** 2))),
        "grad_ground_mu": float(mu_grad[0]),
        "grad_cube_mu": float(mu_grad[1]),
        "grad_mean": float(np.mean(mu_grad[:2])),
        "fit_final_position": pred_np[-1].tolist(),
        "target_final_position": target_np[-1].tolist(),
    }
    tape.zero()
    return result


def main():
    args = parse_args()
    gt_data = load_ground_truth_json(args.gt_json)
    probe_values = args.probe_values or [args.init_value, 0.4, 0.9]
    results = [evaluate_probe(gt_data, value) for value in probe_values]
    nonzero = [r for r in results if abs(r["grad_mean"]) > 1.0e-8]
    out = {
        "system": "slide_cube",
        "fit_param": "friction_coeff",
        "loss_observable": "world position trajectory of the sliding cube center",
        "ground_truth_source": gt_data["path"],
        "steps": gt_data["steps"],
        "dt": gt_data["dt"],
        "solver_iterations": gt_data["solver_iterations"],
        "probes": results,
        "viability": {
            "nonzero_gradient_probe_count": len(nonzero),
            "total_probe_count": len(results),
            "likely_viable_for_friction_sysid": len(nonzero) > 0,
            "note": "This is only a local gradient sanity check around a simple sliding-contact setup.",
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
