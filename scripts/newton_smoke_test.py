#!/usr/bin/env python3
"""Minimal Newton smoke test used for local environment verification.

Run with:
    source /home/yu-tang-liu/.venv/bin/activate
    python scripts/newton_smoke_test.py
"""

import warp as wp
import newton


def main():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
        label="ball",
    )
    builder.add_shape_sphere(body, radius=0.2)

    model = builder.finalize()
    solver = newton.solvers.SolverXPBD(model, iterations=4)
    state0 = model.state()
    state1 = model.state()
    control = model.control()
    contacts = model.contacts()

    for _ in range(20):
        state0.clear_forces()
        model.collide(state0, contacts)
        solver.step(state0, state1, control, contacts, 1.0 / 240.0)
        state0, state1 = state1, state0

    q = state0.body_q.numpy()
    print("body_q shape:", q.shape)
    print("final pose:", q[0].tolist())


if __name__ == "__main__":
    main()
