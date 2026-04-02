#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import warp as wp

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_cli import resolve_config_argv
from slide_cube.util import DEFAULTS, generate_ground_truth, save_ground_truth_json


def parse_args(argv=None):
    argv = resolve_config_argv(argv, script_file=__file__, default_config_name="generate_trajectory_cfg.json")
    p = argparse.ArgumentParser(description="Generate a sliding-cube ground-truth trajectory for contact/friction sysID.")
    p.add_argument("--fit-param", default="friction_coeff", choices=["friction_coeff"])
    p.add_argument("--gt-value", type=float, required=True)
    p.add_argument("--init-height", type=float, default=DEFAULTS["init_height"])
    p.add_argument("--init-x", type=float, default=DEFAULTS["init_x"])
    p.add_argument("--init-vx", type=float, default=DEFAULTS["init_vx"])
    p.add_argument("--half-extent", type=float, default=DEFAULTS["half_extent"])
    p.add_argument("--density", type=float, default=DEFAULTS["density"])
    p.add_argument("--contact-ke", type=float, default=DEFAULTS["contact_ke"])
    p.add_argument("--contact-kd", type=float, default=DEFAULTS["contact_kd"])
    p.add_argument("--contact-kf", type=float, default=DEFAULTS["contact_kf"])
    p.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    p.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    p.add_argument("--solver-iterations", type=int, default=DEFAULTS["solver_iterations"])
    p.add_argument("--output-json", type=str, required=True)
    return p.parse_args(argv)


def main():
    args = parse_args()
    fixed = {
        "init_height": args.init_height,
        "init_x": args.init_x,
        "init_vx": args.init_vx,
        "half_extent": args.half_extent,
        "density": args.density,
        "contact_ke": args.contact_ke,
        "contact_kd": args.contact_kd,
        "contact_kf": args.contact_kf,
    }
    traj, joint_q, joint_qd = generate_ground_truth(
        friction_coeff=args.gt_value,
        steps=args.steps,
        dt=args.dt,
        solver_iterations=args.solver_iterations,
        **fixed,
    )
    save_ground_truth_json(
        Path(args.output_json),
        friction_coeff=args.gt_value,
        fixed=fixed,
        trajectory=traj,
        joint_q=joint_q,
        joint_qd=joint_qd,
        steps=args.steps,
        dt=args.dt,
        solver_iterations=args.solver_iterations,
    )
    print(f"saved ground truth to {args.output_json}")


if __name__ == "__main__":
    wp.init()
    main()
