#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config_cli import resolve_config_argv


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def default_result_json(root: Path) -> Path:
    path = root / "outputs" / "newton_cartpole_batch_sysid" / "result.json"
    if path.exists():
        return path
    raise FileNotFoundError(f"No cartpole batch result json found at {path}")


def main(argv=None):
    argv = resolve_config_argv(
        argv,
        script_file=__file__,
        default_config_name="render_cfg.json",
        config_section="batch",
        positional_keys=("result_json",),
    )
    p = argparse.ArgumentParser(description="Generate cartpole batch summary artifacts and gt/init/fit compare renders beside a batch sysID result.")
    p.add_argument("result_json", type=Path, nargs="?")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--label", type=str, default="cartpole_batch_compare")
    p.add_argument("--subtitle", type=str, default="batch gt vs init vs fit")
    args = p.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    result_json = args.result_json.resolve() if args.result_json is not None else default_result_json(root)
    python = Path(sys.executable)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)

    run([str(python), str(root / "scripts" / "render_sysid_summary.py"), str(result_json)], cwd=root, env=env)
    run(
        [
            str(python),
            str(Path(__file__).with_name("render_compare_cartpole.py")),
            "--result-json",
            str(result_json),
            "--output-dir",
            str(args.output_dir if args.output_dir is not None else (result_json.parent / "compare")),
            "--label",
            args.label,
            "--subtitle",
            args.subtitle,
        ],
        cwd=root,
        env=env,
    )


if __name__ == "__main__":
    main()
