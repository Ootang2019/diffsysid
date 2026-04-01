#!/usr/bin/env python3
from __future__ import annotations

import json

import numpy as np
import warp as wp
import newton

from util import REPO_ROOT, TIP_BODY_INDEX, TIP_LOCAL, URDF_PATH, make_model, point_trajectory_loss, rollout_tip_trajectory, save_tip


def evaluate_probe(param_name: str, gt_value: float, init_value: float, steps: int = 120, dt: float = 1.0 / 240.0):
    gt_kwargs = dict(init_angle_1=0.2, init_angle_2=0.1, init_angvel_1=0.0, init_angvel_2=0.0)
    fit_kwargs = dict(gt_kwargs)

    mapping = {
        "init_angle_1": ("joint_q", 0),
        "init_angle_2": ("joint_q", 1),
        "init_angvel_1": ("joint_qd", 0),
        "init_angvel_2": ("joint_qd", 1),
    }
    if param_name not in mapping:
        raise ValueError(param_name)
    gt_kwargs[param_name] = gt_value
    fit_kwargs[param_name] = init_value
    grad_attr, grad_index = mapping[param_name]

    gt_model = make_model(**gt_kwargs, requires_grad=False)
    target = rollout_tip_trajectory(gt_model, steps=steps, dt=dt, requires_grad=False).numpy()
    target_wp = wp.array(target, dtype=float)

    model = make_model(**fit_kwargs, requires_grad=True)
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    s0 = model.state(requires_grad=True)
    s1 = model.state(requires_grad=True)
    control = model.control(requires_grad=True)
    traj = wp.empty((steps + 1, 3), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        wp.launch(save_tip, dim=1, inputs=[s0.body_q, TIP_BODY_INDEX, TIP_LOCAL, traj, 0])
        for t in range(steps):
            s0.clear_forces()
            solver.step(s0, s1, control, None, dt)
            wp.launch(save_tip, dim=1, inputs=[s1.body_q, TIP_BODY_INDEX, TIP_LOCAL, traj, t + 1])
            s0, s1 = s1, s0
        wp.launch(point_trajectory_loss, dim=(steps + 1), inputs=[traj, target_wp, steps + 1, loss])
    tape.backward(loss)

    grad = getattr(model, grad_attr).grad
    result = {
        "param": param_name,
        "gt_value": gt_value,
        "init_value": init_value,
        "loss": float(loss.numpy()[0]),
        "rmse": float(np.sqrt(np.mean((traj.numpy() - target) ** 2))),
        "grad": None if grad is None else float(grad.numpy()[grad_index]),
        "fit_final_tip_position": traj.numpy()[-1].tolist(),
        "target_final_tip_position": target[-1].tolist(),
    }
    tape.zero()
    return result


def main():
    output_dir = REPO_ROOT / "outputs" / "newton_double_pendulum_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    probes = [
        ("init_angle_1", 0.20, 0.65),
        ("init_angle_2", 0.10, -0.40),
        ("init_angvel_1", 0.00, 1.00),
        ("init_angvel_2", 0.00, -1.00),
    ]
    results = [evaluate_probe(*probe) for probe in probes]
    nonzero = [r for r in results if r["grad"] is not None and abs(r["grad"]) > 1e-10]
    out = {
        "asset": str(URDF_PATH),
        "import_path": "newton.ModelBuilder.add_urdf(...), finalize(...), SolverFeatherstone",
        "loss_observable": "world trajectory of the distal tip point (local point [0,0,-1] on the second link body)",
        "steps": 120,
        "dt": 1.0 / 240.0,
        "probes": results,
        "viability": {
            "nonzero_gradient_probe_count": len(nonzero),
            "total_probe_count": len(results),
            "likely_viable_for_simple_state_sysid": any(r["param"] == "init_angle_1" and r["grad"] is not None and abs(r["grad"]) > 1e-10 for r in results),
        },
    }
    out_path = output_dir / "gradient_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    wp.init()
    main()
