#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def default_result_json(root: Path) -> Path:
    path = root / "outputs" / "newton_double_pendulum_batch_sysid" / "result.json"
    if path.exists():
        return path
    raise FileNotFoundError(f"No double pendulum batch result json found at {path}")


def main():
    p = argparse.ArgumentParser(description="Generate double-pendulum batch summary artifacts and gt/init/fit compare renders beside a batch sysID result.")
    p.add_argument("result_json", type=Path, nargs="?")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[2]
    result_json = args.result_json.resolve() if args.result_json is not None else default_result_json(root)
    python = Path(sys.executable)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)

    run([str(python), str(root / "scripts" / "render_sysid_summary.py"), str(result_json)], cwd=root, env=env)
    run(
        [
            str(python),
            str(Path(__file__).with_name("render_compare_double_pendulum.py")),
            "--result-json",
            str(result_json),
            "--output-dir",
            str(result_json.parent / "compare"),
            "--label",
            "double_pendulum_batch_compare",
            "--subtitle",
            "batch gt vs init vs fit",
        ],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    main()
