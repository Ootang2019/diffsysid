#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
import newton
from PIL import Image

from util import make_model
from diffsysid.render import draw_banner, maybe_make_gif_from_frames, maybe_make_mp4_from_frames


def replay_from_result(result_path: Path, variant: str):
    data = json.loads(result_path.read_text())
    variant_data = data["replay"]["variants"][variant]
    q = np.asarray(variant_data["joint_q"], dtype=np.float32)
    qd = np.asarray(variant_data["joint_qd"], dtype=np.float32)
    time = np.asarray(data["replay"]["time"], dtype=np.float32)
    model = make_model(
        init_angle_1=float(q[0, 0]),
        init_angle_2=float(q[0, 1]),
        init_angvel_1=float(qd[0, 0]),
        init_angvel_2=float(qd[0, 1]),
        requires_grad=False,
    )
    return data, model, q, qd, time


def numeric_rollout(args):
    model = make_model(
        init_angle_1=args.init_angle_1,
        init_angle_2=args.init_angle_2,
        init_angvel_1=args.init_angvel_1,
        init_angvel_2=args.init_angvel_2,
        requires_grad=False,
    )
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    s0 = model.state()
    s1 = model.state()
    control = model.control()
    q = np.empty((args.steps + 1, 2), dtype=np.float32)
    qd = np.empty((args.steps + 1, 2), dtype=np.float32)
    t = np.arange(args.steps + 1, dtype=np.float32) * args.dt
    q[0] = model.joint_q.numpy()
    qd[0] = model.joint_qd.numpy()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for i in range(args.steps):
        s0.clear_forces()
        solver.step(s0, s1, control, None, args.dt)
        q[i + 1] = s1.joint_q.numpy()
        qd[i + 1] = s1.joint_qd.numpy()
        s0, s1 = s1, s0
    return None, model, q, qd, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-json", type=Path)
    p.add_argument("--variant", choices=["gt", "init", "fit"])
    p.add_argument("--init-angle-1", dest="init_angle_1", type=float)
    p.add_argument("--init-angle-2", dest="init_angle_2", type=float)
    p.add_argument("--init-angvel-1", dest="init_angvel_1", type=float)
    p.add_argument("--init-angvel-2", dest="init_angvel_2", type=float)
    p.add_argument("--steps", type=int)
    p.add_argument("--dt", type=float)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--subtitle", type=str, default="")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--cam-x", type=float, default=0.0)
    p.add_argument("--cam-y", type=float, default=-6.0)
    p.add_argument("--cam-z", type=float, default=1.9)
    p.add_argument("--cam-pitch", type=float, default=-11.0)
    p.add_argument("--cam-yaw", type=float, default=90.0)
    args = p.parse_args()

    wp.init()

    if args.result_json is not None:
        if args.variant is None:
            raise SystemExit("--variant is required with --result-json")
        result_data, model, q_traj, qd_traj, t = replay_from_result(args.result_json, args.variant)
        dt = float(t[1] - t[0]) if len(t) > 1 else 1.0 / 60.0
    else:
        required = [args.init_angle_1, args.init_angle_2, args.init_angvel_1, args.init_angvel_2, args.steps, args.dt]
        if any(v is None for v in required):
            raise SystemExit("Numeric rollout requires --init-angle-1 --init-angle-2 --init-angvel-1 --init-angvel-2 --steps --dt")
        result_data, model, q_traj, qd_traj, t = numeric_rollout(args)
        dt = args.dt

    fps = max(1, round(1.0 / dt))
    out_dir = args.output_dir
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(args.cam_x, args.cam_y, args.cam_z), pitch=args.cam_pitch, yaw=args.cam_yaw)
    state = model.state(requires_grad=False)

    for frame_idx, sim_time in enumerate(t.tolist()):
        wp.copy(model.joint_q, wp.array(q_traj[frame_idx], dtype=float))
        wp.copy(model.joint_qd, wp.array(qd_traj[frame_idx], dtype=float))
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        viewer.begin_frame(float(sim_time))
        viewer.log_state(state)
        viewer.end_frame()
        img = Image.fromarray(viewer.get_frame().numpy())
        img = draw_banner(img, f"Newton ViewerGL replay — {args.label}", args.subtitle)
        img.save(frames_dir / f"frame_{frame_idx:04d}.png")

    viewer.close()

    mp4 = maybe_make_mp4_from_frames(frames_dir, out_dir / f"{args.label}.mp4", fps)
    gif = maybe_make_gif_from_frames(frames_dir, out_dir / f"{args.label}.gif", fps)
    manifest = {
        "viewer_backend": "newton.viewer.ViewerGL(headless=True) replayed from explicit joint-state trajectory",
        "label": args.label,
        "variant": args.variant,
        "result_json": None if args.result_json is None else str(args.result_json),
        "steps": int(len(t) - 1),
        "dt": float(dt),
        "fps": fps,
        "initial_state": {
            "joint_q": q_traj[0].tolist(),
            "joint_qd": qd_traj[0].tolist(),
        },
        "artifacts": {
            "frames_dir": str(frames_dir),
            "mp4": None if mp4 is None else str(mp4),
            "gif": None if gif is None else str(gif),
        },
    }
    if result_data is not None:
        manifest["angle_convention"] = result_data.get("angle_convention")
    (out_dir / f"{args.label}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
