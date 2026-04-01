# diffsysid

Minimal differentiable system identification experiments using NVIDIA Newton + Warp.

This repository is a small research benchmark for optimizing physical parameters by backpropagating through Newton simulation trajectories. It covers multiple dynamics scenarios:

- single-link pendulum (`pendulum`)
- articulated double pendulum (`double_pendulum`)
- cartpole and URDF variants (`cartpole`)
- robot arm (`robot_ur10`)

## Project structure

- `diffsysid/`: core reusable Newton parameterization & batch utilities
- `scripts/`: CLI experiment entrypoints
  - `scripts/pendulum/`
  - `scripts/double_pendulum/`
  - `scripts/cartpole/`
  - `scripts/ur10/`
- `outputs/`: run outputs (JSON, PNG, GIF)
- `data/`: optional dataset/storage

## Prerequisites

1. Nvidia Newton Simulator
2. Install project dependencies:

```bash
python3 -m venv .venv
source ~/.venv/bin/activate
cd ~/workspace/diffsysid
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

3. Confirm Warp + newton available:

```bash
source ~/.venv/bin/activate
python -c "import warp, newton; print('warp', warp.__version__)"
```

## Quick usage

Always run from repo root with `PYTHONPATH` and the venv active:

```bash
cd /home/yu-tang-liu/workspace/diffsysid
source ~/.venv/bin/activate
PYTHONPATH=. python scripts/pendulum/newton_pendulum_sysid.py --fit-param init_angle --gt-value 0.2 --init-value 0.65 --iters 120 --output-json outputs/newton_pendulum_sysid/result.json
```

### Pendulum probe example

```bash
PYTHONPATH=. python scripts/pendulum/probe_pendulum_gradients.py
```

### Render result

```bash
PYTHONPATH=. python scripts/pendulum/render_newton_pendulum.py --result-json outputs/newton_pendulum_sysid/result.json --variant fit --label pendulum_fit --output-dir outputs/newton_pendulum_render
```

## Script locations

### pendulum
- `scripts/pendulum/newton_pendulum_sysid.py`
- `scripts/pendulum/probe_pendulum_gradients.py`
- `scripts/pendulum/render_newton_pendulum.py`

### double_pendulum
- `scripts/double_pendulum/newton_double_pendulum_sysid.py`
- `scripts/double_pendulum/probe_double_pendulum_gradients.py`
- `scripts/double_pendulum/render_newton_double_pendulum.py`

### spring
- `scripts/spring/newton_spring_sysid.py`
- `scripts/spring/newton_spring_batch_sysid.py`

### cartpole
- `scripts/cartpole/newton_cartpole_sysid.py`
- `scripts/cartpole/probe_cartpole_gradients.py`
- `scripts/cartpole/render_newton_cartpole.py`

### robot UR10
- `scripts/robot_ur10/newton_robot_ur10_sysid.py`

## Batch results handling

- `final_fit.best_params` or `final_fit.best_joint_q` indicates the selected best replica parameters.
- `final_fit.traj_rmse` is trajectory error using selected params.
- `history` contains per-iteration stats for convergence analysis.

## Contact

This is a lab experiment repository. Tune script arguments for your scenario. Always validate with the JSON `final_fit` values before trusting visualizations.
