from __future__ import annotations

import json
from pathlib import Path

import newton
import numpy as np
import warp as wp



REPO_ROOT = Path(__file__).resolve().parents[2]
URDF_PATH = REPO_ROOT / "data" / "urdf" / "slide_cube.urdf"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "slide_cube"
DATA_ROOT = REPO_ROOT / "data" / "gt_datasets"
BODY_INDEX = 0
GROUND_SHAPE_INDEX = 0
CUBE_SHAPE_INDICES = (1, 2)
PARAM_SPECS = {
    "friction_coeff": (("shape_material_mu", GROUND_SHAPE_INDEX), ("shape_material_mu", CUBE_SHAPE_INDICES[0]), ("shape_material_mu", CUBE_SHAPE_INDICES[1])),
}
NONNEGATIVE_PARAM_NAMES = {"friction_coeff"}
DEFAULTS = {
    "friction_coeff": 0.25,
    "init_height": 0.101,
    "init_x": 0.0,
    "init_vx": 1.5,
    "half_extent": 0.1,
    "density": 500.0,
    "steps": 60,
    "dt": 1.0 / 240.0,
    "solver_iterations": 16,
    "contact_ke": 1.0e5,
    "contact_kd": 1.0e3,
    "contact_kf": 1.0e3,
}


@wp.kernel
def save_box_position(body_q: wp.array(dtype=wp.transform), body_index: int, out: wp.array2d(dtype=float), step: int):
    if wp.tid() == 0:
        p = wp.transform_get_translation(body_q[body_index])
        out[step, 0] = p[0]
        out[step, 1] = p[1]
        out[step, 2] = p[2]


@wp.kernel
def point_mse_loss(pred: wp.array2d(dtype=float), target: wp.array2d(dtype=float), steps: int, loss: wp.array(dtype=float)):
    tid = wp.tid()
    if tid < steps:
        dx = pred[tid, 0] - target[tid, 0]
        dy = pred[tid, 1] - target[tid, 1]
        dz = pred[tid, 2] - target[tid, 2]
        wp.atomic_add(loss, 0, (dx * dx + dy * dy + dz * dz) / float(steps * 3))


@wp.kernel
def set_array_value(arr: wp.array(dtype=float), index: int, value: float):
    if wp.tid() == 0:
        arr[index] = value


def display_param(name: str, raw_value: float, angle_mode: str) -> float:
    del name, angle_mode
    return float(raw_value)


def _assign_contact_params(model: newton.Model, *, friction_coeff: float, contact_ke: float, contact_kd: float, contact_kf: float) -> None:
    for index in (GROUND_SHAPE_INDEX, *CUBE_SHAPE_INDICES):
        wp.launch(set_array_value, dim=1, inputs=[model.shape_material_mu, index, float(friction_coeff)])
        wp.launch(set_array_value, dim=1, inputs=[model.shape_material_ke, index, float(contact_ke)])
        wp.launch(set_array_value, dim=1, inputs=[model.shape_material_kd, index, float(contact_kd)])
        wp.launch(set_array_value, dim=1, inputs=[model.shape_material_kf, index, float(contact_kf)])


def make_model(
    *,
    friction_coeff: float = DEFAULTS["friction_coeff"],
    init_height: float = DEFAULTS["init_height"],
    init_x: float = DEFAULTS["init_x"],
    init_vx: float = DEFAULTS["init_vx"],
    half_extent: float = DEFAULTS["half_extent"],
    density: float = DEFAULTS["density"],
    contact_ke: float = DEFAULTS["contact_ke"],
    contact_kd: float = DEFAULTS["contact_kd"],
    contact_kf: float = DEFAULTS["contact_kf"],
    requires_grad: bool = False,
):
    if abs(float(half_extent) - DEFAULTS["half_extent"]) > 1.0e-9:
        raise ValueError("slide_cube URDF currently assumes half_extent=0.1 (box size 0.2 m).")
    expected_mass = 8.0 * float(half_extent) ** 3 * float(density)
    if abs(expected_mass - 4.0) > 1.0e-6:
        raise ValueError("slide_cube URDF currently assumes density=500.0 with half_extent=0.1 (mass 4.0 kg).")

    builder = newton.ModelBuilder()
    builder.add_ground_plane(label="ground")
    builder.add_urdf(
        str(URDF_PATH),
        floating=True,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    model = builder.finalize(requires_grad=requires_grad)
    wp.copy(model.joint_q, wp.array(np.array([float(init_x), 0.0, float(init_height), 0.0, 0.0, 0.0, 1.0], dtype=np.float32), dtype=float))
    wp.copy(model.joint_qd, wp.array(np.array([float(init_vx), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), dtype=float))
    _assign_contact_params(
        model,
        friction_coeff=friction_coeff,
        contact_ke=contact_ke,
        contact_kd=contact_kd,
        contact_kf=contact_kf,
    )
    return model


def apply_friction_coeff(model: newton.Model, value: float) -> None:
    for arr_name, index in PARAM_SPECS["friction_coeff"]:
        wp.launch(set_array_value, dim=1, inputs=[getattr(model, arr_name), index, float(value)])


def rollout_state_trajectory(model: newton.Model, *, steps: int, dt: float, solver_iterations: int, requires_grad: bool):
    solver = newton.solvers.SolverXPBD(model, iterations=solver_iterations)
    s0 = model.state(requires_grad=requires_grad)
    s1 = model.state(requires_grad=requires_grad)
    control = model.control(requires_grad=requires_grad)
    contacts = model.contacts()
    traj = wp.empty((steps + 1, 3), dtype=float, requires_grad=requires_grad)
    body_q_traj = np.empty((steps + 1, 1, 7), dtype=np.float32)
    body_qd_traj = np.empty((steps + 1, 1, 6), dtype=np.float32)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    wp.launch(save_box_position, dim=1, inputs=[s0.body_q, BODY_INDEX, traj, 0])
    body_q_traj[0] = s0.body_q.numpy().reshape(1, 7)
    body_qd_traj[0] = s0.body_qd.numpy().reshape(1, 6)
    for t in range(steps):
        s0.clear_forces()
        model.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        wp.launch(save_box_position, dim=1, inputs=[s1.body_q, BODY_INDEX, traj, t + 1])
        body_q_traj[t + 1] = s1.body_q.numpy().reshape(1, 7)
        body_qd_traj[t + 1] = s1.body_qd.numpy().reshape(1, 6)
        s0, s1 = s1, s0
    return traj, body_q_traj, body_qd_traj


def rollout_position_trajectory(model: newton.Model, *, steps: int, dt: float, solver_iterations: int, requires_grad: bool):
    traj, _, _ = rollout_state_trajectory(model, steps=steps, dt=dt, solver_iterations=solver_iterations, requires_grad=requires_grad)
    return traj


def generate_ground_truth(*, friction_coeff: float, steps: int, dt: float, solver_iterations: int, **fixed):
    model = make_model(friction_coeff=friction_coeff, requires_grad=False, **fixed)
    traj, joint_q, joint_qd = rollout_state_trajectory(model, steps=steps, dt=dt, solver_iterations=solver_iterations, requires_grad=False)
    return traj.numpy(), joint_q, joint_qd


def save_ground_truth_json(path: Path, *, friction_coeff: float, fixed: dict, trajectory: np.ndarray, joint_q: np.ndarray, joint_qd: np.ndarray, steps: int, dt: float, solver_iterations: int) -> dict:
    payload = {
        "type": "ground_truth_trajectory",
        "system": "slide_cube",
        "asset": str(URDF_PATH),
        "fit_param": "friction_coeff",
        "fit_params": ["friction_coeff"],
        "ground_truth": {
            "friction_coeff": float(friction_coeff),
            "friction_coeff_raw": float(friction_coeff),
        },
        "fixed_parameters": {**{k: float(v) for k, v in fixed.items()}, **{f"{k}_raw": float(v) for k, v in fixed.items()}},
        "angle_convention": {"mode": "identity", "raw_zero_meaning": "not angle-based", "raw_upright_angle": None},
        "config": {
            "steps": int(steps),
            "dt": float(dt),
            "solver_iterations": int(solver_iterations),
            "angle_mode": "identity",
        },
        "loss_observable": "world position trajectory of the sliding cube center",
        "replay": {
            "time": (np.arange(steps + 1, dtype=np.float32) * dt).tolist(),
            "variants": {
                "gt": {
                    "body_q": np.asarray(joint_q, dtype=np.float32).tolist(),
                    "body_qd": np.asarray(joint_qd, dtype=np.float32).tolist(),
                }
            },
        },
        "target_trajectory": np.asarray(trajectory, dtype=np.float32).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load_ground_truth_json(path: str | Path) -> dict:
    path = Path(path)
    raw = json.loads(path.read_text())
    gt_variant = raw["replay"]["variants"]["gt"]
    body_q = np.asarray(gt_variant.get("body_q", gt_variant.get("joint_q")), dtype=np.float32)
    body_qd = np.asarray(gt_variant.get("body_qd", gt_variant.get("joint_qd")), dtype=np.float32)
    fixed = {}
    for key, value in raw.get("fixed_parameters", {}).items():
        if not key.endswith("_raw"):
            fixed[key] = float(value)
    return {
        "path": str(path),
        "data": raw,
        "fit_params": ["friction_coeff"],
        "ground_truth_raw": {"friction_coeff": float(raw["ground_truth"].get("friction_coeff_raw", raw["ground_truth"]["friction_coeff"]))},
        "fixed": fixed,
        "target_trajectory": np.asarray(raw["target_trajectory"], dtype=np.float32),
        "gt_q": body_q,
        "gt_qd": body_qd,
        "time": np.asarray(raw["replay"]["time"], dtype=np.float32),
        "steps": int(raw["config"]["steps"]),
        "dt": float(raw["config"]["dt"]),
        "solver_iterations": int(raw["config"].get("solver_iterations", DEFAULTS["solver_iterations"])),
    }
