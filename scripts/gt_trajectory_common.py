from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_param_payload(
    fit_params: list[str],
    raw_values: np.ndarray,
    *,
    display_value_fn,
    angle_mode: str,
) -> dict:
    payload = {}
    for name, raw_value in zip(fit_params, raw_values):
        payload[name] = display_value_fn(name, raw_value, angle_mode)
        payload[f"{name}_raw"] = float(raw_value)
    return payload


def build_fixed_payload(
    fixed: dict[str, float],
    *,
    display_value_fn,
    angle_mode: str,
) -> dict:
    payload = {}
    for name, raw_value in fixed.items():
        payload[name] = display_value_fn(name, raw_value, angle_mode)
        payload[f"{name}_raw"] = float(raw_value)
    return payload


def save_ground_truth_json(
    output_json: Path,
    *,
    asset: str,
    fit_params: list[str],
    gt_values_raw: np.ndarray,
    fixed: dict[str, float],
    steps: int,
    dt: float,
    target_trajectory: np.ndarray,
    joint_q: np.ndarray,
    joint_qd: np.ndarray,
    angle_mode: str,
    angle_convention: dict,
    loss_observable: str,
    display_value_fn,
) -> dict:
    result = {
        "type": "ground_truth_trajectory",
        "asset": asset,
        "fit_param": fit_params[0] if len(fit_params) == 1 else None,
        "fit_params": fit_params,
        "ground_truth": build_param_payload(fit_params, gt_values_raw, display_value_fn=display_value_fn, angle_mode=angle_mode),
        "fixed_parameters": build_fixed_payload(fixed, display_value_fn=display_value_fn, angle_mode=angle_mode),
        "angle_convention": angle_convention,
        "loss_observable": loss_observable,
        "config": {
            "steps": int(steps),
            "dt": float(dt),
            "angle_mode": angle_mode,
        },
        "replay": {
            "time": (np.arange(steps + 1, dtype=np.float32) * dt).tolist(),
            "variants": {
                "gt": {
                    "joint_q": joint_q.tolist(),
                    "joint_qd": joint_qd.tolist(),
                }
            },
        },
        "target_trajectory": np.asarray(target_trajectory, dtype=np.float32).tolist(),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2))
    return result


def load_ground_truth_json(path: str | Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text())

    target_trajectory = np.asarray(data["target_trajectory"], dtype=np.float32)
    gt_variant = data["replay"]["variants"]["gt"]
    gt_q = np.asarray(gt_variant["joint_q"], dtype=np.float32)
    gt_qd = np.asarray(gt_variant["joint_qd"], dtype=np.float32)
    time = np.asarray(data["replay"]["time"], dtype=np.float32)
    dt = float(time[1] - time[0]) if time.shape[0] >= 2 else float(data.get("config", {}).get("dt", 0.0))

    fit_params = list(data.get("fit_params") or ([data["fit_param"]] if data.get("fit_param") else []))
    ground_truth_raw = {}
    for name in fit_params:
        raw_key = f"{name}_raw"
        if raw_key in data.get("ground_truth", {}):
            ground_truth_raw[name] = float(data["ground_truth"][raw_key])
        elif name in data.get("ground_truth", {}):
            ground_truth_raw[name] = float(data["ground_truth"][name])

    fixed_raw = {}
    for key, value in data.get("fixed_parameters", {}).items():
        if key.endswith("_raw"):
            fixed_raw[key[:-4]] = float(value)

    return {
        "path": str(path),
        "data": data,
        "fit_params": fit_params,
        "ground_truth_raw": ground_truth_raw,
        "fixed_raw": fixed_raw,
        "target_trajectory": target_trajectory,
        "gt_q": gt_q,
        "gt_qd": gt_qd,
        "time": time,
        "steps": int(target_trajectory.shape[0] - 1),
        "dt": dt,
    }
