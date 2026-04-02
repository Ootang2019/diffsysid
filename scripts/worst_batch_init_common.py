#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np

from gt_trajectory_common import load_ground_truth_json


def _fit_params(data: dict) -> list[str]:
    fit_params = data.get("fit_params")
    if fit_params:
        return list(fit_params)
    fit_param = data.get("fit_param")
    if fit_param is None:
        raise KeyError("Result JSON must contain fit_param or fit_params.")
    return [fit_param]


def _load_backend(result_json: Path, data: dict):
    scripts_root = Path(__file__).resolve().parent
    asset_name = Path(data["asset"]).name
    if asset_name == "cartpole.urdf":
        cartpole_dir = str(scripts_root / "cartpole")
        if cartpole_dir not in sys.path:
            sys.path.insert(0, cartpole_dir)
        from cartpole.newton_cartpole_sysid import display_param, make_model_with_params, rollout_joint_state_trajectory
        from cartpole.util import rollout_tip_trajectory

        return display_param, make_model_with_params, rollout_joint_state_trajectory, rollout_tip_trajectory
    if asset_name == "pendulum.urdf":
        pendulum_dir = str(scripts_root / "pendulum")
        if pendulum_dir not in sys.path:
            sys.path.insert(0, pendulum_dir)
        from pendulum.newton_pendulum_sysid import display_param, make_model_with_params, rollout_joint_state_trajectory
        from pendulum.util import rollout_tip_trajectory

        return display_param, make_model_with_params, rollout_joint_state_trajectory, rollout_tip_trajectory
    if asset_name == "double_pendulum.urdf":
        double_pendulum_dir = str(scripts_root / "double_pendulum")
        if double_pendulum_dir not in sys.path:
            sys.path.insert(0, double_pendulum_dir)
        from double_pendulum.newton_double_pendulum_sysid import display_value, make_model_with_params, rollout_joint_state_trajectory
        from double_pendulum.util import rollout_tip_trajectory

        return display_value, make_model_with_params, rollout_joint_state_trajectory, rollout_tip_trajectory
    raise ValueError(f"Unsupported asset for worst-initial selection: {asset_name} ({result_json})")


def maybe_apply_worst_initial(data: dict, *, result_json: Path) -> dict:
    initial_population = data.get("initial_population")
    gt_source = data.get("ground_truth_source")
    if initial_population is None or gt_source is None:
        return data

    display_param, make_model_with_params, rollout_joint_state_trajectory, rollout_tip_trajectory = _load_backend(result_json, data)
    gt_data = load_ground_truth_json(gt_source)
    fit_params = _fit_params(data)
    fixed = dict(gt_data["fixed_raw"])
    steps = int(gt_data["steps"])
    dt = float(gt_data["dt"])
    target = np.asarray(gt_data["target_trajectory"], dtype=np.float32)
    angle_mode = data.get("angle_convention", {}).get("mode", "urdf_raw")

    worst_idx = None
    worst_loss = -np.inf
    worst_rmse = None
    worst_pred = None
    worst_q = None
    worst_qd = None
    worst_raw_values = None

    for env_index, raw_values in enumerate(initial_population["raw_params"]):
        raw_array = np.asarray(raw_values, dtype=np.float64)
        model, _ = make_model_with_params(fit_params, raw_array, fixed, requires_grad=False)
        pred = rollout_tip_trajectory(model, steps=steps, dt=dt, requires_grad=False).numpy()
        loss = float(np.mean((pred - target) ** 2))
        if loss > worst_loss:
            q, qd = rollout_joint_state_trajectory(model, steps, dt)
            worst_idx = env_index
            worst_loss = loss
            worst_rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
            worst_pred = pred
            worst_q = q
            worst_qd = qd
            worst_raw_values = raw_array

    if worst_idx is None:
        return data

    updated = copy.deepcopy(data)
    initial_guess = dict(updated.get("initial_guess", {}))
    initial_guess["env_index"] = int(worst_idx)
    initial_guess["loss"] = float(worst_loss)
    initial_guess["rmse"] = float(worst_rmse)
    for name, raw_value in zip(fit_params, worst_raw_values):
        initial_guess[f"{name}_raw"] = float(raw_value)
        initial_guess[name] = float(display_param(name, float(raw_value), angle_mode))
    updated["initial_guess"] = initial_guess
    updated["initial_prediction"] = worst_pred.tolist()
    updated.setdefault("replay", {}).setdefault("variants", {})["init"] = {
        "joint_q": worst_q.tolist(),
        "joint_qd": worst_qd.tolist(),
    }
    return updated
