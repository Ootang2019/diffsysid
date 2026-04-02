#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gt_trajectory_common import save_ground_truth_json
from newton_double_pendulum_sysid import (
    FIT_PARAM_CHOICES,
    FIT_PARAM_SELECTION_CHOICES,
    angle_convention_payload,
    build_fixed_parameters,
    display_value,
    make_model_with_params,
    parameter_to_raw,
    rollout_joint_state_trajectory,
    rollout_tip_trajectory,
    sample_random_init_values,
)
from util import DYNAMIC_PARAM_NAMES
from util import URDF_PATH


def parse_args():
    p = argparse.ArgumentParser(description="Generate and save a double-pendulum ground-truth trajectory for later sysID runs.")
    p.add_argument("--fit-param", choices=FIT_PARAM_SELECTION_CHOICES)
    p.add_argument("--gt-value", type=str)
    p.add_argument("--fit-params", nargs="+")
    p.add_argument("--gt-values", nargs="+")
    p.add_argument("--init-value", type=float, default=0.0)
    p.add_argument("--init-values", nargs="+", type=float)
    p.add_argument("--init-angle-1", dest="init_angle_1", type=float, default=0.2)
    p.add_argument("--init-angle-2", dest="init_angle_2", type=float, default=0.1)
    p.add_argument("--angle-mode", choices=["urdf_raw", "top_offset"], default="urdf_raw")
    p.add_argument("--init-angvel-1", dest="init_angvel_1", type=float, default=0.0)
    p.add_argument("--init-angvel-2", dest="init_angvel_2", type=float, default=0.0)
    p.add_argument("--joint1-armature", type=float, default=0.0)
    p.add_argument("--joint1-stiffness", type=float, default=0.0)
    p.add_argument("--joint1-damping", type=float, default=0.0)
    p.add_argument("--joint2-armature", type=float, default=0.0)
    p.add_argument("--joint2-stiffness", type=float, default=0.0)
    p.add_argument("--joint2-damping", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--dt", type=float, default=1.0 / 240.0)
    p.add_argument("--random-gt-span", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, required=True)
    return p.parse_args()


def resolve_generator_request(args):
    fit_params = args.fit_params if args.fit_params is not None else ([args.fit_param] if args.fit_param is not None else None)
    gt_values = args.gt_values if args.gt_values is not None else ([args.gt_value] if args.gt_value is not None else None)
    if fit_params == ["all"]:
        fit_params = list(FIT_PARAM_CHOICES)
    if fit_params == ["all_dynamics"]:
        fit_params = [name for name in FIT_PARAM_CHOICES if name in DYNAMIC_PARAM_NAMES]
    if fit_params is None or gt_values is None:
        raise ValueError("Specify either --fit-param/--gt-value or --fit-params/--gt-values.")
    if len(set(fit_params)) != len(fit_params):
        raise ValueError("Duplicate fit params are not allowed.")
    fixed = build_fixed_parameters(args)
    if len(gt_values) == 1 and gt_values[0] == "random":
        centers = np.asarray([display_value(name, fixed[name], args.angle_mode) for name in fit_params], dtype=np.float64)
        return list(fit_params), sample_random_init_values(
            list(fit_params),
            centers,
            random_init_span=args.random_gt_span,
            seed=args.seed,
        )
    if len(fit_params) != len(gt_values):
        raise ValueError("fit params and gt values must have the same length, or use --gt-values random.")
    return list(fit_params), np.asarray([float(value) for value in gt_values], dtype=np.float64)


def main():
    args = parse_args()
    fit_params, gt_values_in = resolve_generator_request(args)
    gt_values_raw = np.asarray([parameter_to_raw(name, value, args.angle_mode) for name, value in zip(fit_params, gt_values_in)], dtype=np.float64)
    fixed = build_fixed_parameters(args)

    model, _ = make_model_with_params(fit_params, gt_values_raw, fixed, requires_grad=False)
    target_traj = rollout_tip_trajectory(model, steps=args.steps, dt=args.dt, requires_grad=False).numpy()
    joint_q, joint_qd = rollout_joint_state_trajectory(model, args.steps, args.dt)
    save_ground_truth_json(
        output_json=Path(args.output_json),
        asset=str(URDF_PATH),
        fit_params=fit_params,
        gt_values_raw=gt_values_raw,
        fixed=fixed,
        steps=args.steps,
        dt=args.dt,
        target_trajectory=target_traj,
        joint_q=joint_q,
        joint_qd=joint_qd,
        angle_mode=args.angle_mode,
        angle_convention=angle_convention_payload(args),
        loss_observable="world trajectory of the distal tip point (local point [0,0,-1] on the second link body)",
        display_value_fn=display_value,
    )
    print(f"saved ground truth to {args.output_json}")


if __name__ == "__main__":
    wp.init()
    main()
