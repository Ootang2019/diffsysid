from __future__ import annotations

from pathlib import Path

import newton
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = REPO_ROOT / "data" / "urdf" / "pendulum.urdf"
TIP_BODY_INDEX = 0
TIP_LOCAL = wp.vec3(0.0, 0.0, -1.0)
PARAM_SPECS = {
    "init_angle": ("joint_q", 0),
    "init_angvel": ("joint_qd", 0),
    "hinge_armature": ("joint_armature", 0),
    "hinge_stiffness": ("joint_target_ke", 0),
    "hinge_damping": ("joint_target_kd", 0),
}
DYNAMIC_PARAM_NAMES = {"hinge_armature", "hinge_stiffness", "hinge_damping"}
NONNEGATIVE_PARAM_NAMES = set(DYNAMIC_PARAM_NAMES)


@wp.kernel
def point_trajectory_loss(pred: wp.array2d(dtype=float), target: wp.array2d(dtype=float), steps: int, loss: wp.array(dtype=float)):
    tid = wp.tid()
    if tid < steps:
        dx = pred[tid, 0] - target[tid, 0]
        dy = pred[tid, 1] - target[tid, 1]
        dz = pred[tid, 2] - target[tid, 2]
        wp.atomic_add(loss, 0, dx * dx + dy * dy + dz * dz)


@wp.kernel
def set_array_value(arr: wp.array(dtype=float), index: int, value: float):
    if wp.tid() == 0:
        arr[index] = value


@wp.kernel
def save_tip(body_q: wp.array(dtype=wp.transform), body_index: int, local_tip: wp.vec3, out: wp.array2d(dtype=float), step: int):
    if wp.tid() == 0:
        p = wp.transform_point(body_q[body_index], local_tip)
        out[step, 0] = p[0]
        out[step, 1] = p[1]
        out[step, 2] = p[2]


def apply_model_parameters(model: newton.Model, params: dict[str, float]) -> None:
    for name, value in params.items():
        arr_name, index = PARAM_SPECS[name]
        wp.launch(set_array_value, dim=1, inputs=[getattr(model, arr_name), index, float(value)])


def make_model(
    *,
    init_angle: float = 0.2,
    init_angvel: float = 0.0,
    hinge_armature: float = 0.0,
    hinge_stiffness: float = 0.0,
    hinge_damping: float = 0.0,
    requires_grad: bool = False,
):
    builder = newton.ModelBuilder()
    builder.add_urdf(
        str(URDF_PATH),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    model = builder.finalize(requires_grad=requires_grad)
    apply_model_parameters(
        model,
        {
            "init_angle": init_angle,
            "init_angvel": init_angvel,
            "hinge_armature": hinge_armature,
            "hinge_stiffness": hinge_stiffness,
            "hinge_damping": hinge_damping,
        },
    )
    return model


def rollout_tip_trajectory(model: newton.Model, steps: int, dt: float, requires_grad: bool):
    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    s0 = model.state(requires_grad=requires_grad)
    s1 = model.state(requires_grad=requires_grad)
    control = model.control(requires_grad=requires_grad)
    traj = wp.empty((steps + 1, 3), dtype=float, requires_grad=requires_grad)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    wp.launch(save_tip, dim=1, inputs=[s0.body_q, TIP_BODY_INDEX, TIP_LOCAL, traj, 0])
    for t in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        wp.launch(save_tip, dim=1, inputs=[s1.body_q, TIP_BODY_INDEX, TIP_LOCAL, traj, t + 1])
        s0, s1 = s1, s0
    return traj
