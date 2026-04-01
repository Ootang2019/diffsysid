# diffsysid

Differentiable system identification with NVIDIA Newton + Warp.

A research benchmark for optimizing physical parameters by backpropagating through Newton simulation trajectories. Covers pendulum dynamics, articulated multi-body systems, elasticity, and cartpole control tasks.

## Project structure

```
diffsysid/
├── diffsysid/               # Core reusable batch utilities
│   ├── batch.py            # Elite restart, population metrics
│   ├── basic_pendulum.py    # Articulated pendulum builder
│   ├── spring.py           # Spring-damper systems
│   ├── common.py           # Softplus, loss kernels
│   ├── io.py               # JSON result handling
│   └── __init__.py
├── scripts/
│   ├── batch_sysid_common.py    # Shared batch experiment code
│   ├── pendulum/                # Single & double pendulum experiments
│   ├── basic_pendulum/          # Articulated 2-DOF pendulum
│   ├── spring/                  # Spring-damper parameter ID
│   └── cartpole/                # URDF cartpole from external assets
├── outputs/                     # Results (JSON, PNG, GIF, MP4)
├── data/
│   ├── raw/                     # Raw test data
│   ├── processed/               # Processed datasets
│   ├── urdf/                    # URDF model definitions
│   └── sysid_batch/             # Batch config snapshots
├── .venv/                       # Python 3.12 virtual environment (local)
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and set up environment

```bash
git clone git@github.com:ootang2019/diffsysid.git
cd diffsysid
source ./.venv/bin/activate  # local venv already included
python -c "import warp as wp; print('warp', wp.__version__)"
```

If you clone fresh (without `.venv/`):

```bash
python3 -m venv .venv
source ./.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify Warp + Newton

```bash
python -c "import warp as wp, newton; print('Warp:', wp.__version__)"
```

## Quick start

All commands run from repo root with `PYTHONPATH=.` and venv active.

### Pendulum sysID (single bob, identify spring length)

```bash
source ./.venv/bin/activate
cd ~/workspace/diffsysid

PYTHONPATH=. python scripts/pendulum/newton_pendulum_sysid.py \
  --system single \
  --gt-length 1.0 \
  --init-length 0.8 \
  --iters 100 \
  --output-json outputs/newton_pendulum_sysid/example.json

# Render the result
PYTHONPATH=. python scripts/pendulum/render_newton_pendulum_sysid.py \
  outputs/newton_pendulum_sysid/example.json
```

### Batch sysID (multi-replica pendulum)

```bash
PYTHONPATH=. python scripts/pendulum/newton_pendulum_batch_sysid.py \
  --system single \
  --fit-param length \
  --env-count 8 \
  --iters 50 \
  --output-json outputs/newton_pendulum_sysid/batch_example.json
```

### Spring damper system

```bash
PYTHONPATH=. python scripts/spring/newton_spring_sysid.py \
  --gt-stiffness 80.0 \
  --gt-damping 6.0 \
  --init-stiffness 100.0 \
  --init-damping 4.0 \
  --iters 80 \
  --output-json outputs/newton_spring_sysid/example.json

# Render
PYTHONPATH=. python scripts/spring/render_newton_spring_sysid.py \
  outputs/newton_spring_sysid/example.json
```

### Articulated pendulum (2-body chain)

```bash
PYTHONPATH=. python scripts/basic_pendulum/newton_basic_pendulum_sysid.py \
  --system double \
  --gt-length1 1.0 --gt-length2 1.0 \
  --init-length1 0.9 --init-length2 0.9 \
  --iters 100 \
  --output-json outputs/newton_basic_pendulum_sysid/example.json

# Batch version
PYTHONPATH=. python scripts/basic_pendulum/newton_basic_pendulum_batch_sysid.py \
  --env-count 8 --iters 50 \
  --output-json outputs/newton_basic_pendulum_sysid/batch_example.json
```

### Cartpole (URDF-based)

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_sysid.py \
  --fit-param init_pole_angle \
  --gt-value 0.10 \
  --init-value 1.00 \
  --iters 80 \
  --output-json outputs/newton_cartpole_sysid/example.json

# Render summary and animation
PYTHONPATH=. python scripts/cartpole/render_newton_cartpole_sysid.py \
  outputs/newton_cartpole_sysid/example.json \
  --summary-png outputs/newton_cartpole_sysid/summary.png \
  --gif outputs/newton_cartpole_sysid/animation.gif \
  --mp4 outputs/newton_cartpole_sysid/animation.mp4

# ViewerGL replay
PYTHONPATH=. python scripts/cartpole/render_newton_cartpole.py \
  --result-json outputs/newton_cartpole_sysid/example.json \
  --variant fit --label cartpole_fit \
  --output-dir outputs/newton_cartpole_sysid/viewer
```

## Key features

### Batch system ID
- Multi-start optimization with `env_count` parallel replicas
- Elite restart strategy: weak replicas cloned from best + noise
- Per-environment loss tracking and population statistics
- Shared ground-truth parameters across all environments

### Rendering pipeline
- Static PNG summaries with geometry overlays and convergence plots
- Animated GIFs and MP4s for trajectory comparison
- JSON metadata preserving angle conventions and fit selection logic

### Supported systems
| System | File | Fit params | Notes |
|--------|------|-----------|-------|
| Pendulum (single) | `pendulum/newton_pendulum_sysid.py` | `length`, `init_angle` | Spring-tethered bob |
| Pendulum (batch) | `pendulum/newton_pendulum_batch_sysid.py` | `length`, `init_angle` | Multi-replica with elite restart |
| Basic pendulum | `basic_pendulum/newton_basic_pendulum_sysid.py` | `length1`, `length2`, `init_angle` | Two-body articulated chain |
| Basic pendulum (batch) | `basic_pendulum/newton_basic_pendulum_batch_sysid.py` | `length1`, `length2` | Batched multi-start |
| Spring | `spring/newton_spring_sysid.py` | `stiffness`, `damping` | Single particle on spring |
| Spring (batch) | `spring/newton_spring_batch_sysid.py` | `stiffness`, `damping` | Multi-replica spring fitting |
| Cartpole | `cartpole/newton_cartpole_sysid.py` | `init_pole_angle`, etc. | URDF-imported dynamics |

## Batch result interpretation

JSON output includes:

- `config`: all run hyperparameters
- `ground_truth`: hidden parameters used to generate target trajectory
- `initial_guess`: per-environment starting parameters
- `final_fit`: solved parameters + loss + metrics
  - `best_params`: winning replica parameters
  - `traj_rmse`: final trajectory RMSE
  - `stiffness_param_rmse` / `distance_to_gt_*`: parameter error vs ground truth
  - `population_std`: population spread at convergence
- `history`: per-iteration convergence log
- `restart_events`: elite clone and random restart events

## Troubleshooting

**CUDA/Warp initialization fails:**
```bash
python -c "import warp as wp; wp.init()"
```
Ensure NVIDIA drivers are installed and CUDA 12+ is available.

**Import errors for `diffsysid` or `batch_sysid_common`:**
- Always use `PYTHONPATH=. python ...` when running scripts
- All scripts must be run from repo root

**Visualization missing fonts:**
Your system must have DejaVuSans or Arial fonts installed. On Linux:
```bash
sudo apt install fonts-dejavu fonts-liberation
```

## References

- NVIDIA Newton docs: https://docs.nvidia.com/nsight-systems/profiling-nvidia-newton/
- Warp: https://github.com/NVIDIA/warp
- Differentiable physics: Ajay et al. (2016), Liang et al. (2019)

## License

See LICENSE file.

## Contact

Lab experiment repository for research. Tune hyperparameters (`--iters`, `--lr`, `--env-count`) for your scenario. Always inspect JSON `final_fit` before relying on rendered outputs.
