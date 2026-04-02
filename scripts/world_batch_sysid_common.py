from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
import warp as wp


@dataclass
class WorldBatchResult:
    losses: np.ndarray
    rmses: np.ndarray
    grads: np.ndarray
    trajectories: np.ndarray
    world_losses: np.ndarray
    world_rmses: np.ndarray
    world_grads: np.ndarray
    world_trajectories: np.ndarray


@wp.kernel
def save_world_tip(
    body_q: wp.array(dtype=wp.transform),
    body_world_start: wp.array(dtype=int),
    local_body_index: int,
    local_tip: wp.vec3,
    out: wp.array3d(dtype=float),
    step: int,
):
    world = wp.tid()
    body_index = body_world_start[world] + local_body_index
    p = wp.transform_point(body_q[body_index], local_tip)
    out[world, step, 0] = p[0]
    out[world, step, 1] = p[1]
    out[world, step, 2] = p[2]


@wp.kernel
def world_point_mse_loss(
    pred: wp.array3d(dtype=float),
    target: wp.array3d(dtype=float),
    point_counts: wp.array(dtype=int),
    loss: wp.array(dtype=float),
):
    world, point = wp.tid()
    count = point_counts[world]
    if point < count:
        dx = pred[world, point, 0] - target[world, point, 0]
        dy = pred[world, point, 1] - target[world, point, 1]
        dz = pred[world, point, 2] - target[world, point, 2]
        wp.atomic_add(loss, world, (dx * dx + dy * dy + dz * dz) / float(count * 3))


@wp.kernel
def sum_world_losses(losses: wp.array(dtype=float), total: wp.array(dtype=float)):
    world = wp.tid()
    wp.atomic_add(total, 0, losses[world])


def build_replicated_urdf_model(
    urdf_path: str | Path,
    *,
    world_count: int,
    requires_grad: bool,
):
    sub_builder = newton.ModelBuilder()
    sub_builder.add_urdf(
        str(urdf_path),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    builder = newton.ModelBuilder()
    builder.replicate(sub_builder, world_count=world_count, spacing=(0.0, 0.0, 0.0))
    return builder.finalize(requires_grad=requires_grad)


def _world_start_arrays(model: newton.Model) -> dict[str, np.ndarray]:
    joint_coord_world_start = model.joint_coord_world_start.numpy()
    joint_dof_world_start = model.joint_dof_world_start.numpy()
    return {
        "joint_q": joint_coord_world_start,
        "joint_qd": joint_dof_world_start,
        "joint_armature": joint_dof_world_start,
        "joint_target_ke": joint_dof_world_start,
        "joint_target_kd": joint_dof_world_start,
    }


def assign_world_parameters(
    model: newton.Model,
    *,
    param_specs: dict[str, tuple[str, int]],
    fit_params: list[str],
    world_param_values: np.ndarray,
    world_fixed_params: list[dict[str, float]],
) -> None:
    arrays = {}
    for arr_name, _ in param_specs.values():
        if arr_name not in arrays:
            arrays[arr_name] = getattr(model, arr_name).numpy().copy()

    world_starts = _world_start_arrays(model)
    for world_index, fixed in enumerate(world_fixed_params):
        combined = dict(fixed)
        for param_name, param_value in zip(fit_params, world_param_values[world_index]):
            combined[param_name] = float(param_value)
        for param_name, param_value in combined.items():
            arr_name, local_index = param_specs[param_name]
            arrays[arr_name][world_starts[arr_name][world_index] + local_index] = float(param_value)

    for arr_name, arr_np in arrays.items():
        dst = getattr(model, arr_name)
        wp.copy(dst, wp.array(arr_np, dtype=float, device=dst.device))


def evaluate_world_batch(
    *,
    urdf_path: str | Path,
    param_specs: dict[str, tuple[str, int]],
    fit_params: list[str],
    per_env_param_values: np.ndarray,
    world_to_env: np.ndarray,
    world_fixed_params: list[dict[str, float]],
    target_batch_np: np.ndarray,
    point_counts: np.ndarray,
    dt: float,
    tip_body_index: int,
    tip_local,
) -> WorldBatchResult:
    env_count = int(per_env_param_values.shape[0])
    world_count = int(len(world_fixed_params))
    fit_count = len(fit_params)
    max_points = int(target_batch_np.shape[1])
    max_steps = max_points - 1

    model = build_replicated_urdf_model(urdf_path, world_count=world_count, requires_grad=True)
    world_param_values = np.asarray(per_env_param_values, dtype=np.float64)[np.asarray(world_to_env, dtype=np.int32)]
    assign_world_parameters(
        model,
        param_specs=param_specs,
        fit_params=fit_params,
        world_param_values=world_param_values,
        world_fixed_params=world_fixed_params,
    )

    solver = newton.solvers.SolverFeatherstone(model, angular_damping=0.0)
    state0 = model.state(requires_grad=True)
    state1 = model.state(requires_grad=True)
    control = model.control(requires_grad=True)

    pred = wp.empty((world_count, max_points, 3), dtype=float, requires_grad=True)
    target = wp.array(np.asarray(target_batch_np, dtype=np.float32), dtype=float)
    point_counts_wp = wp.array(np.asarray(point_counts, dtype=np.int32), dtype=int)
    world_losses = wp.zeros(world_count, dtype=float, requires_grad=True)
    total_loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
        newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
        wp.launch(
            save_world_tip,
            dim=world_count,
            inputs=[state0.body_q, model.body_world_start, tip_body_index, tip_local, pred, 0],
        )
        for step in range(max_steps):
            state0.clear_forces()
            solver.step(state0, state1, control, None, dt)
            wp.launch(
                save_world_tip,
                dim=world_count,
                inputs=[state1.body_q, model.body_world_start, tip_body_index, tip_local, pred, step + 1],
            )
            state0, state1 = state1, state0
        wp.launch(world_point_mse_loss, dim=(world_count, max_points), inputs=[pred, target, point_counts_wp, world_losses])
        wp.launch(sum_world_losses, dim=world_count, inputs=[world_losses, total_loss])
    tape.backward(total_loss)

    pred_np = pred.numpy()
    world_losses_np = world_losses.numpy().astype(np.float64)
    target_np = np.asarray(target_batch_np, dtype=np.float64)
    point_counts_np = np.asarray(point_counts, dtype=np.int32)

    world_rmses = np.empty(world_count, dtype=np.float64)
    for world_index in range(world_count):
        valid = int(point_counts_np[world_index])
        diff = pred_np[world_index, :valid] - target_np[world_index, :valid]
        world_rmses[world_index] = float(np.sqrt(np.mean(diff * diff)))

    world_starts = _world_start_arrays(model)
    world_grads = np.empty((world_count, fit_count), dtype=np.float64)
    for fit_index, param_name in enumerate(fit_params):
        arr_name, local_index = param_specs[param_name]
        grad_np = getattr(model, arr_name).grad.numpy()
        starts = world_starts[arr_name]
        for world_index in range(world_count):
            world_grads[world_index, fit_index] = float(grad_np[starts[world_index] + local_index])

    losses = np.zeros(env_count, dtype=np.float64)
    rmses = np.zeros(env_count, dtype=np.float64)
    grads = np.zeros((env_count, fit_count), dtype=np.float64)
    trajectories = np.zeros((env_count, max_points, 3), dtype=np.float64)
    total_weights = np.zeros(env_count, dtype=np.float64)

    for world_index, env_index in enumerate(np.asarray(world_to_env, dtype=np.int32)):
        weight = float(point_counts_np[world_index])
        total_weights[env_index] += weight
        losses[env_index] += world_losses_np[world_index] * weight
        rmses[env_index] += (world_rmses[world_index] ** 2) * weight
        grads[env_index] += world_grads[world_index] * weight

    for env_index in range(env_count):
        weight = max(total_weights[env_index], 1.0)
        losses[env_index] /= weight
        rmses[env_index] = float(np.sqrt(rmses[env_index] / weight))
        grads[env_index] /= weight

    # For the standard non-snippet case there is one world per env, so this is exact.
    # For snippet batching the aggregate replay is still produced elsewhere from the selected fit.
    for world_index, env_index in enumerate(np.asarray(world_to_env, dtype=np.int32)):
        if total_weights[env_index] == float(point_counts_np[world_index]):
            trajectories[env_index] = pred_np[world_index]

    tape.zero()
    return WorldBatchResult(
        losses=losses,
        rmses=rmses,
        grads=grads,
        trajectories=trajectories,
        world_losses=world_losses_np,
        world_rmses=world_rmses,
        world_grads=world_grads,
        world_trajectories=pred_np.astype(np.float64),
    )
