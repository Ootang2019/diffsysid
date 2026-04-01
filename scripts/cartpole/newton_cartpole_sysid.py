#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp
import newton

from util import POLE_BODY_INDEX, POLE_TIP_LOCAL, URDF_PATH, make_model, rollout_tip_trajectory, save_pole_tip, set_array_value


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


@dataclass
class IterationLog:
    iteration: int
    loss: float
    rmse: float
    param_value: float
    grad: float


def make_model_with_param(param_name: str, param_value: float, fixed: dict[str, float], requires_grad: bool):
    model = make_model(
        init_cart_pos=fixed["init_cart_pos"],
        init_pole_angle=fixed["init_pole_angle"],
        init_cart_vel=fixed["init_cart_vel"],
        init_pole_angvel=fixed["init_pole_angvel"],
        requires_grad=requires_grad,
    )
    if param_name == "init_pole_angle":
        wp.launch(set_array_value, dim=1, inputs=[model.joint_q, 1, float(param_value)])
        grad_attr, grad_index = "joint_q", 1
    elif param_name == "init_cart_pos":
        wp.launch(set_array_value, dim=1, inputs=[model.joint_q, 0, float(param_value)])
        grad_attr, grad_index = "joint_q", 0
    elif param_name == "init_cart_vel":
        wp.launch(set_array_value, dim=1, inputs=[model.joint_qd, 0, float(param_value)])
        grad_attr, grad_index = "joint_qd", 0
    elif param_name == "init_pole_angvel":
        wp.launch(set_array_value, dim=1, inputs=[model.joint_qd, 1, float(param_value)])
        grad_attr, grad_index = "joint_qd", 1
    else:
        raise ValueError(param_name)
    return model, grad_attr, grad_index


def evaluate(target_traj: wp.array, param_name: str, param_value: float, fixed: dict[str, float], steps: int, dt: float):
    model, grad_attr, grad_index = make_model_with_param(param_name, param_value, fixed, requires_grad=True)
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    s0 = model.state(requires_grad=True)
    s1 = model.state(requires_grad=True)
    control = model.control(requires_grad=True)

    pred = wp.empty((steps + 1, 3), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        wp.launch(save_pole_tip, dim=1, inputs=[s0.body_q, POLE_BODY_INDEX, POLE_TIP_LOCAL, pred, 0])
        for t in range(steps):
            s0.clear_forces()
            solver.step(s0, s1, control, None, dt)
            wp.launch(save_pole_tip, dim=1, inputs=[s1.body_q, POLE_BODY_INDEX, POLE_TIP_LOCAL, pred, t + 1])
            s0, s1 = s1, s0
        wp.launch(point_mse_loss, dim=(steps + 1), inputs=[pred, target_traj, steps + 1, loss])
    tape.backward(loss)

    pred_np = pred.numpy()
    target_np = target_traj.numpy()
    grad_np = getattr(model, grad_attr).grad.numpy()
    out = {
        "loss": float(loss.numpy()[0]),
        "rmse": float(np.sqrt(np.mean((pred_np - target_np) ** 2))),
        "grad": float(grad_np[grad_index]),
        "trajectory": pred_np,
    }
    tape.zero()
    return out


def run(args):
    gt_value_raw = top_offset_to_raw_angle(args.gt_value) if args.angle_mode == "top_offset" and args.fit_param == "init_pole_angle" else args.gt_value
    init_value_raw = top_offset_to_raw_angle(args.init_value) if args.angle_mode == "top_offset" and args.fit_param == "init_pole_angle" else args.init_value
    init_pole_angle_raw = top_offset_to_raw_angle(args.init_pole_angle) if args.angle_mode == "top_offset" else args.init_pole_angle

    fixed = {
        "init_cart_pos": args.init_cart_pos,
        "init_pole_angle": gt_value_raw if args.fit_param == "init_pole_angle" else init_pole_angle_raw,
        "init_cart_vel": args.init_cart_vel,
        "init_pole_angvel": args.init_pole_angvel,
    }
    gt_model, _, _ = make_model_with_param(args.fit_param, gt_value_raw, fixed, requires_grad=False)
    gt_traj_np = rollout_tip_trajectory(gt_model, steps=args.steps, dt=args.dt, requires_grad=False).numpy()
    gt_traj = wp.array(gt_traj_np, dtype=float)

    param_value = init_value_raw
    m = 0.0
    v = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    history: list[IterationLog] = []
    initial_metrics = evaluate(gt_traj, args.fit_param, param_value, fixed, args.steps, args.dt)
    best = {"param_value": param_value, **initial_metrics}

    for it in range(args.iters):
        metrics = evaluate(gt_traj, args.fit_param, param_value, fixed, args.steps, args.dt)
        history.append(IterationLog(it, metrics["loss"], metrics["rmse"], param_value, metrics["grad"]))
        print(f"iter={it:03d} loss={metrics['loss']:.6e} rmse={metrics['rmse']:.6e} {args.fit_param}={param_value:.6f} grad={metrics['grad']:.6e}")
        if metrics["loss"] < best["loss"]:
            best = {"param_value": param_value, **metrics}
        g = float(np.clip(metrics["grad"], -args.grad_clip, args.grad_clip))
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1 ** (it + 1))
        v_hat = v / (1.0 - beta2 ** (it + 1))
        param_value -= args.lr * m_hat / (np.sqrt(v_hat) + eps)

    final_metrics = evaluate(gt_traj, args.fit_param, param_value, fixed, args.steps, args.dt)
    if final_metrics["loss"] > best["loss"]:
        final_param_value = best["param_value"]
        final_metrics = evaluate(gt_traj, args.fit_param, final_param_value, fixed, args.steps, args.dt)
        selection = "best_seen"
    else:
        final_param_value = param_value
        selection = "last_iter"

    def display_angle(raw_value: float) -> float:
        return raw_angle_to_top_offset(raw_value) if args.angle_mode == "top_offset" else raw_value

    gt_replay_model, _, _ = make_model_with_param(args.fit_param, gt_value_raw, fixed, requires_grad=False)
    init_replay_model, _, _ = make_model_with_param(args.fit_param, init_value_raw, fixed, requires_grad=False)
    fit_replay_model, _, _ = make_model_with_param(args.fit_param, final_param_value, fixed, requires_grad=False)
    gt_q, gt_qd = rollout_joint_state_trajectory(gt_replay_model, args.steps, args.dt)
    init_q, init_qd = rollout_joint_state_trajectory(init_replay_model, args.steps, args.dt)
    fit_q, fit_qd = rollout_joint_state_trajectory(fit_replay_model, args.steps, args.dt)

    fixed_display = dict(fixed)
    fixed_display["init_pole_angle_raw"] = fixed["init_pole_angle"]
    fixed_display["init_pole_angle"] = display_angle(fixed["init_pole_angle"])

    result = {
        "asset": str(URDF_PATH),
        "import_path": "newton.ModelBuilder.add_urdf(...), finalize(...), SolverFeatherstone",
        "loss_observable": "world trajectory of the pole tip point (local point [0,0,-1] on pole body)",
        "fit_param": args.fit_param,
        "angle_convention": {
            "mode": args.angle_mode,
            "raw_zero_meaning": "pole hanging straight down",
            "raw_upright_angle": UPRIGHT_RAW_ANGLE,
            "top_offset_definition": "when mode=top_offset, reported angle = pi - raw_urdf_angle, so 0 means upright/top and positive values tilt toward negative world x",
        },
        "ground_truth": {args.fit_param: display_angle(gt_value_raw), f"{args.fit_param}_raw": gt_value_raw},
        "fixed_parameters": fixed_display,
        "initial_guess": {args.fit_param: display_angle(init_value_raw), f"{args.fit_param}_raw": init_value_raw, "loss": initial_metrics["loss"], "rmse": initial_metrics["rmse"]},
        "final_fit": {args.fit_param: display_angle(final_param_value), f"{args.fit_param}_raw": final_param_value, "loss": final_metrics["loss"], "rmse": final_metrics["rmse"], "grad": final_metrics["grad"], "selection": selection},
        "best_seen": {args.fit_param: display_angle(best["param_value"]), f"{args.fit_param}_raw": best["param_value"], "loss": best["loss"], "rmse": best["rmse"]},
        "config": vars(args),
        "replay": {
            "time": (np.arange(args.steps + 1, dtype=np.float32) * args.dt).tolist(),
            "variants": {
                "gt": {"joint_q": gt_q.tolist(), "joint_qd": gt_qd.tolist()},
                "init": {"joint_q": init_q.tolist(), "joint_qd": init_qd.tolist()},
                "fit": {"joint_q": fit_q.tolist(), "joint_qd": fit_qd.tolist()},
            },
        },
        "history": [asdict(x) for x in history],
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
    p = argparse.ArgumentParser(description="URDF-imported cartpole sysID on one simple state parameter.")
    p.add_argument("--fit-param", choices=["init_pole_angle", "init_cart_pos", "init_cart_vel", "init_pole_angvel"], required=True)
    p.add_argument("--gt-value", type=float, required=True)
    p.add_argument("--init-value", type=float, required=True)
    p.add_argument("--init-cart-pos", type=float, default=0.0)
    p.add_argument("--init-pole-angle", type=float, default=0.2)
    p.add_argument("--angle-mode", choices=["urdf_raw", "top_offset"], default="urdf_raw")
    p.add_argument("--init-cart-vel", type=float, default=0.0)
    p.add_argument("--init-pole-angvel", type=float, default=0.0)
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
