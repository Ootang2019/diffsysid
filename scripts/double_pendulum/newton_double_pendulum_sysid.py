#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp
import newton

from util import (
    PARAM_SPECS,
    TIP_BODY_INDEX,
    TIP_LOCAL,
    URDF_PATH,
    make_model,
    rollout_tip_trajectory,
    save_tip,
)


UPRIGHT_RAW_ANGLE = float(np.pi)


def top_offset_to_raw_angle(top_offset: float) -> float:
    return UPRIGHT_RAW_ANGLE - float(top_offset)


def raw_angle_to_top_offset(raw_angle: float) -> float:
    return UPRIGHT_RAW_ANGLE - float(raw_angle)


def rollout_joint_state_trajectory(model: newton.Model, steps: int, dt: float):
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    s0 = model.state(requires_grad=False)
    s1 = model.state(requires_grad=False)
    control = model.control(requires_grad=False)

    q_traj = np.empty((steps + 1, int(model.joint_q.shape[0])), dtype=np.float32)
    qd_traj = np.empty((steps + 1, int(model.joint_qd.shape[0])), dtype=np.float32)

    q_traj[0] = model.joint_q.numpy()
    qd_traj[0] = model.joint_qd.numpy()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for t in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        q_traj[t + 1] = s1.joint_q.numpy()
        qd_traj[t + 1] = s1.joint_qd.numpy()
        s0, s1 = s1, s0
    return q_traj, qd_traj


@wp.kernel
def point_mse_loss(pred: wp.array2d(dtype=float), target: wp.array2d(dtype=float), steps: int, loss: wp.array(dtype=float)):
    tid = wp.tid()
    if tid < steps:
        dx = pred[tid, 0] - target[tid, 0]
        dy = pred[tid, 1] - target[tid, 1]
        dz = pred[tid, 2] - target[tid, 2]
        wp.atomic_add(loss, 0, (dx * dx + dy * dy + dz * dz) / float(steps * 3))

FIT_PARAM_CHOICES = tuple(PARAM_SPECS.keys())
ANGLE_PARAMS = {"init_angle_1", "init_angle_2"}


def resolve_fit_request(args):
    fit_params = args.fit_params if args.fit_params is not None else ([args.fit_param] if args.fit_param is not None else None)
    gt_values = args.gt_values if args.gt_values is not None else ([args.gt_value] if args.gt_value is not None else None)
    init_values = args.init_values if args.init_values is not None else ([args.init_value] if args.init_value is not None else None)
    if fit_params is None or gt_values is None or init_values is None:
        raise ValueError("Specify either --fit-param/--gt-value/--init-value or --fit-params/--gt-values/--init-values.")
    if not (len(fit_params) == len(gt_values) == len(init_values)):
        raise ValueError("fit params, gt values, and init values must have the same length.")
    if len(set(fit_params)) != len(fit_params):
        raise ValueError("Duplicate fit params are not allowed.")
    return list(fit_params), np.asarray(gt_values, dtype=np.float64), np.asarray(init_values, dtype=np.float64)


def parameter_to_raw(name: str, value: float, angle_mode: str) -> float:
    if angle_mode == "top_offset" and name in ANGLE_PARAMS:
        return top_offset_to_raw_angle(value)
    return float(value)


def display_value(name: str, raw_value: float, angle_mode: str) -> float:
    if angle_mode == "top_offset" and name in ANGLE_PARAMS:
        return raw_angle_to_top_offset(raw_value)
    return float(raw_value)


def make_model_with_params(fit_params: list[str], param_values: np.ndarray, fixed: dict[str, float], requires_grad: bool):
    params = dict(fixed)
    for name, value in zip(fit_params, param_values):
        params[name] = float(value)
    model = make_model(requires_grad=requires_grad, **params)
    grad_refs = [PARAM_SPECS[name] for name in fit_params]
    return model, grad_refs


def evaluate(target_traj: wp.array, fit_params: list[str], param_values: np.ndarray, fixed: dict[str, float], steps: int, dt: float):
    model, grad_refs = make_model_with_params(fit_params, param_values, fixed, requires_grad=True)
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    s0 = model.state(requires_grad=True)
    s1 = model.state(requires_grad=True)
    control = model.control(requires_grad=True)

    pred = wp.empty((steps + 1, 3), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        wp.launch(save_tip, dim=1, inputs=[s0.body_q, TIP_BODY_INDEX, TIP_LOCAL, pred, 0])
        for t in range(steps):
            s0.clear_forces()
            solver.step(s0, s1, control, None, dt)
            wp.launch(save_tip, dim=1, inputs=[s1.body_q, TIP_BODY_INDEX, TIP_LOCAL, pred, t + 1])
            s0, s1 = s1, s0
        wp.launch(point_mse_loss, dim=(steps + 1), inputs=[pred, target_traj, steps + 1, loss])
    tape.backward(loss)

    pred_np = pred.numpy()
    target_np = target_traj.numpy()
    grads: dict[str, float] = {}
    for name, (grad_attr, grad_index) in zip(fit_params, grad_refs):
        grads[name] = float(getattr(model, grad_attr).grad.numpy()[grad_index])
    out = {
        "loss": float(loss.numpy()[0]),
        "rmse": float(np.sqrt(np.mean((pred_np - target_np) ** 2))),
        "grads": grads,
        "trajectory": pred_np,
    }
    if len(fit_params) == 1:
        out["grad"] = grads[fit_params[0]]
    tape.zero()
    return out


def run(args):
    fit_params, gt_values_in, init_values_in = resolve_fit_request(args)
    gt_values_raw = np.asarray([parameter_to_raw(name, value, args.angle_mode) for name, value in zip(fit_params, gt_values_in)], dtype=np.float64)
    init_values_raw = np.asarray([parameter_to_raw(name, value, args.angle_mode) for name, value in zip(fit_params, init_values_in)], dtype=np.float64)

    fixed = {
        "init_angle_1": parameter_to_raw("init_angle_1", args.init_angle_1, args.angle_mode),
        "init_angle_2": parameter_to_raw("init_angle_2", args.init_angle_2, args.angle_mode),
        "init_angvel_1": args.init_angvel_1,
        "init_angvel_2": args.init_angvel_2,
        "joint1_armature": args.joint1_armature,
        "joint1_stiffness": args.joint1_stiffness,
        "joint1_damping": args.joint1_damping,
        "joint2_armature": args.joint2_armature,
        "joint2_stiffness": args.joint2_stiffness,
        "joint2_damping": args.joint2_damping,
    }

    gt_model, _ = make_model_with_params(fit_params, gt_values_raw, fixed, requires_grad=False)
    gt_traj_np = rollout_tip_trajectory(gt_model, steps=args.steps, dt=args.dt, requires_grad=False).numpy()
    gt_traj = wp.array(gt_traj_np, dtype=float)

    param_values = init_values_raw.copy()
    m = np.zeros_like(param_values)
    v = np.zeros_like(param_values)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    history: list[dict] = []
    initial_metrics = evaluate(gt_traj, fit_params, param_values, fixed, args.steps, args.dt)
    best = {"param_values": param_values.copy(), **initial_metrics}

    for it in range(args.iters):
        metrics = evaluate(gt_traj, fit_params, param_values, fixed, args.steps, args.dt)
        history_entry = {
            "iteration": it,
            "loss": metrics["loss"],
            "rmse": metrics["rmse"],
            "param_values": {name: float(value) for name, value in zip(fit_params, param_values)},
            "grads": metrics["grads"],
        }
        if len(fit_params) == 1:
            history_entry["param_value"] = float(param_values[0])
            history_entry["grad"] = float(metrics["grads"][fit_params[0]])
        history.append(history_entry)
        status = " ".join(f"{name}={value:.6f}" for name, value in zip(fit_params, param_values))
        grad_status = " ".join(f"{name}_grad={metrics['grads'][name]:.6e}" for name in fit_params)
        print(f"iter={it:03d} loss={metrics['loss']:.6e} rmse={metrics['rmse']:.6e} {status} {grad_status}")
        if metrics["loss"] < best["loss"]:
            best = {"param_values": param_values.copy(), **metrics}
        g = np.clip(np.asarray([metrics["grads"][name] for name in fit_params], dtype=np.float64), -args.grad_clip, args.grad_clip)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** (it + 1))
        v_hat = v / (1.0 - beta2 ** (it + 1))
        param_values -= args.lr * m_hat / (np.sqrt(v_hat) + eps)

    final_metrics = evaluate(gt_traj, fit_params, param_values, fixed, args.steps, args.dt)
    if final_metrics["loss"] > best["loss"]:
        final_param_values = best["param_values"].copy()
        final_metrics = evaluate(gt_traj, fit_params, final_param_values, fixed, args.steps, args.dt)
        selection = "best_seen"
    else:
        final_param_values = param_values.copy()
        selection = "last_iter"

    gt_replay_model, _ = make_model_with_params(fit_params, gt_values_raw, fixed, requires_grad=False)
    init_replay_model, _ = make_model_with_params(fit_params, init_values_raw, fixed, requires_grad=False)
    fit_replay_model, _ = make_model_with_params(fit_params, final_param_values, fixed, requires_grad=False)
    gt_q, gt_qd = rollout_joint_state_trajectory(gt_replay_model, args.steps, args.dt)
    init_q, init_qd = rollout_joint_state_trajectory(init_replay_model, args.steps, args.dt)
    fit_q, fit_qd = rollout_joint_state_trajectory(fit_replay_model, args.steps, args.dt)

    fixed_display = {}
    for name, value in fixed.items():
        if name in fit_params:
            continue
        fixed_display[f"{name}_raw"] = value
        fixed_display[name] = display_value(name, value, args.angle_mode)

    def pack_param_section(raw_values: np.ndarray, *, include_metrics: dict | None = None, include_selection: str | None = None):
        payload = {}
        for name, raw_value in zip(fit_params, raw_values):
            payload[name] = display_value(name, raw_value, args.angle_mode)
            payload[f"{name}_raw"] = float(raw_value)
        if include_metrics is not None:
            payload["loss"] = include_metrics["loss"]
            payload["rmse"] = include_metrics["rmse"]
            if len(fit_params) == 1:
                payload["grad"] = include_metrics["grads"][fit_params[0]]
            else:
                payload["grads"] = include_metrics["grads"]
        if include_selection is not None:
            payload["selection"] = include_selection
        return payload

    result = {
        "asset": str(URDF_PATH),
        "import_path": "newton.ModelBuilder.add_urdf(...), finalize(...), SolverFeatherstone",
        "loss_observable": "world trajectory of the distal tip point (local point [0,0,-1] on the second link body)",
        "fit_param": fit_params[0] if len(fit_params) == 1 else None,
        "fit_params": fit_params,
        "angle_convention": {
            "mode": args.angle_mode,
            "raw_zero_meaning": "link hanging straight down",
            "raw_upright_angle": UPRIGHT_RAW_ANGLE,
            "top_offset_definition": "when mode=top_offset, reported angle = pi - raw_urdf_angle, so 0 means upright/top",
        },
        "ground_truth": pack_param_section(gt_values_raw),
        "fixed_parameters": fixed_display,
        "initial_guess": pack_param_section(init_values_raw, include_metrics=initial_metrics),
        "final_fit": pack_param_section(final_param_values, include_metrics=final_metrics, include_selection=selection),
        "best_seen": pack_param_section(best["param_values"], include_metrics=best),
        "config": vars(args),
        "replay": {
            "time": (np.arange(args.steps + 1, dtype=np.float32) * args.dt).tolist(),
            "variants": {
                "gt": {"joint_q": gt_q.tolist(), "joint_qd": gt_qd.tolist()},
                "init": {"joint_q": init_q.tolist(), "joint_qd": init_qd.tolist()},
                "fit": {"joint_q": fit_q.tolist(), "joint_qd": fit_qd.tolist()},
            },
        },
        "history": history,
        "target_trajectory": gt_traj_np.tolist(),
        "initial_prediction": initial_metrics["trajectory"].tolist(),
        "final_prediction": final_metrics["trajectory"].tolist(),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved results to {out}")
    return result


def parse_args():
    p = argparse.ArgumentParser(description="URDF-imported double-pendulum sysID on one simple state parameter.")
    p.add_argument("--fit-param", choices=FIT_PARAM_CHOICES)
    p.add_argument("--gt-value", type=float)
    p.add_argument("--init-value", type=float)
    p.add_argument("--fit-params", nargs="+", choices=FIT_PARAM_CHOICES)
    p.add_argument("--gt-values", nargs="+", type=float)
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
    p.add_argument("--iters", type=int, default=80)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1e4)
    p.add_argument("--output-json", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    wp.init()
    run(parse_args())
