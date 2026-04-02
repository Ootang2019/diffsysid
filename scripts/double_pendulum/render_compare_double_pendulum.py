#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from worst_batch_init_common import maybe_apply_worst_initial


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def maybe_prepare_worst_init_result(result_json: Path) -> Path:
    data = json.loads(result_json.read_text())
    updated = maybe_apply_worst_initial(data, result_json=result_json)
    if updated is data:
        return result_json

    tmp_dir = Path(tempfile.mkdtemp(prefix="double_pendulum_batch_compare_"))
    tmp_json = tmp_dir / result_json.name
    tmp_json.write_text(json.dumps(updated, indent=2))
    return tmp_json


def main():
    p = argparse.ArgumentParser(description="Render gt/init/fit double-pendulum replays and stitch them into one comparison gif/mp4.")
    p.add_argument("--result-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--label", type=str, default="double_pendulum_compare")
    p.add_argument("--subtitle", type=str, default="")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=540)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    python = Path(sys.executable)
    render_script = Path(__file__).with_name("render_newton_double_pendulum.py")
    stitch_script = root / "scripts" / "stitch_triptych_frames.py"
    output_dir = args.output_dir
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    render_result_json = maybe_prepare_worst_init_result(args.result_json)

    view_dirs = {
        "gt": output_dir / "gt",
        "init": output_dir / "init",
        "fit": output_dir / "fit",
    }

    for variant, view_dir in view_dirs.items():
        run(
            [
                str(python),
                str(render_script),
                "--result-json",
                str(render_result_json),
                "--variant",
                variant,
                "--label",
                variant,
                "--subtitle",
                args.subtitle or f"double pendulum {variant} replay",
                "--output-dir",
                str(view_dir),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
            ],
            cwd=root,
            env=env,
        )

    run(
        [
            str(python),
            str(stitch_script),
            "--frames-dirs",
            str(view_dirs["gt"] / "frames"),
            str(view_dirs["init"] / "frames"),
            str(view_dirs["fit"] / "frames"),
            "--labels",
            "Ground Truth",
            "Initial",
            "Fitted",
            "--output-dir",
            str(output_dir / "compare"),
            "--title",
            "Double Pendulum SysID Comparison",
            "--subtitle",
            args.subtitle or args.result_json.name,
            "--fps",
            str(args.fps),
            "--stem",
            args.label,
        ],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    main()
