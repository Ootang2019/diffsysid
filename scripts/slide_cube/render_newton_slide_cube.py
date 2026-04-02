#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import warp as wp
import newton

from diffsysid.render import draw_banner, maybe_make_gif_from_frames, maybe_make_mp4_from_frames
from util import DEFAULTS, make_model


def replay_from_result(result_path: Path, variant: str):
    data = json.loads(result_path.read_text())
    variant_data = data["replay"]["variants"][variant]
    fixed = dict(DEFAULTS)
    fixed.update(data.get("fixed_parameters", {}))
    if variant == "gt":
        friction = float(data["ground_truth"]["friction_coeff_raw"])
    elif variant == "init":
        friction = float(data["initial_guess"]["friction_coeff_raw"])
    else:
        friction = float(data["final_fit"]["friction_coeff_raw"])
    model = make_model(
        friction_coeff=friction,
        init_height=float(fixed["init_height"]),
        init_x=float(fixed["init_x"]),
        init_vx=float(fixed["init_vx"]),
        half_extent=float(fixed["half_extent"]),
        density=float(fixed["density"]),
        contact_ke=float(fixed["contact_ke"]),
        contact_kd=float(fixed["contact_kd"]),
        contact_kf=float(fixed["contact_kf"]),
        requires_grad=False,
    )
    body_q = variant_data["body_q"] if "body_q" in variant_data else variant_data["joint_q"]
    body_qd = variant_data["body_qd"] if "body_qd" in variant_data else variant_data["joint_qd"]
    return data, model, body_q, body_qd, data["replay"]["time"]


def main():
    p = argparse.ArgumentParser(description="Render gt/init/fit slide_cube replays from saved joint trajectories.")
    p.add_argument("--result-json", type=Path, required=True)
    p.add_argument("--variant", choices=["gt", "init", "fit"], required=True)
    p.add_argument("--label", type=str, required=True)
    p.add_argument("--subtitle", type=str, default="")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--cam-x", type=float, default=1.4)
    p.add_argument("--cam-y", type=float, default=-2.0)
    p.add_argument("--cam-z", type=float, default=1.0)
    p.add_argument("--cam-pitch", type=float, default=-12.0)
    p.add_argument("--cam-yaw", type=float, default=70.0)
    args = p.parse_args()

    wp.init()
    data, model, body_q_traj, body_qd_traj, t = replay_from_result(args.result_json, args.variant)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0 / 60.0
    fps = max(1, round(1.0 / dt))
    out_dir = args.output_dir
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(args.cam_x, args.cam_y, args.cam_z), pitch=args.cam_pitch, yaw=args.cam_yaw)
    state = model.state(requires_grad=False)

    for frame_idx, sim_time in enumerate(t):
        wp.copy(state.body_q, wp.array(body_q_traj[frame_idx], dtype=wp.transform))
        wp.copy(state.body_qd, wp.array(body_qd_traj[frame_idx], dtype=wp.spatial_vector))
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
        "result_json": str(args.result_json),
        "steps": int(len(t) - 1),
        "dt": float(dt),
        "fps": fps,
        "artifacts": {
            "frames_dir": str(frames_dir),
            "mp4": None if mp4 is None else str(mp4),
            "gif": None if gif is None else str(gif),
        },
        "angle_convention": data.get("angle_convention"),
    }
    (out_dir / f"{args.label}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
