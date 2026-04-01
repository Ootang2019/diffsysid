# diffsysid

Differentiable system identification with NVIDIA Newton + Warp.

This repo currently focuses on three URDF-backed systems:
- cartpole
- pendulum
- double pendulum

Each system supports:
- single-run sysid
- batch sysid with multi-start restarts
- summary generation
- side-by-side `gt | init | fit` Newton renders

## Project structure

```text
diffsysid/
├── diffsysid/                         # Shared optimization and batch utilities
├── data/
│   └── urdf/
│       ├── cartpole.urdf
│       ├── pendulum.urdf
│       └── double_pendulum.urdf
├── scripts/
│   ├── batch_sysid_common.py         # Shared Newton sysid helpers
│   ├── render_sysid_summary.py       # Shared summary JSON/PNG generator
│   ├── stitch_triptych_frames.py     # Shared frame stitching helper
│   ├── cartpole/
│   ├── pendulum/
│   └── double_pendulum/
├── outputs/                           # Result JSON, summary PNG, GIF, MP4
├── .venv/                             # Repo-local Python environment
├── requirements.txt
└── README.md
```

## Setup

Run everything from the repo root.

```bash
git clone git@github.com:ootang2019/diffsysid.git
cd diffsysid
source ./.venv/bin/activate
PYTHONPATH=. python -c "import warp as wp, newton; print('Warp:', wp.__version__)"
```

If you need to create the environment:

```bash
python3 -m venv .venv
source ./.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Conventions used throughout this repo:
- run commands from the repo root
- set `PYTHONPATH=.`
- prefer `python -m pip` over `pip`

## Quick start

### 1. Single-run sysid

Cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_sysid.py \
  --fit-param init_pole_angle \
  --gt-value 0.10 \
  --init-value 0.80 \
  --iters 80 \
  --output-json outputs/newton_cartpole_sysid/result.json
```

Pendulum:

```bash
PYTHONPATH=. python scripts/pendulum/newton_pendulum_sysid.py \
  --fit-param init_angle \
  --gt-value 0.20 \
  --init-value 0.60 \
  --iters 80 \
  --output-json outputs/newton_pendulum_sysid/result.json
```

Double pendulum:

```bash
PYTHONPATH=. python scripts/double_pendulum/newton_double_pendulum_sysid.py \
  --fit-param init_angle_2 \
  --gt-value 0.10 \
  --init-value -0.40 \
  --iters 80 \
  --output-json outputs/newton_double_pendulum_sysid/result.json
```

### 2. Single-run summary and compare render

Each per-robot `render_sysid_summary.py` wrapper does both:
- generates `summary.json`
- generates `summary.png`
- renders `gt`, `init`, and `fit`
- stitches them into a compare GIF and MP4 under `compare/`

Cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/render_sysid_summary.py \
  outputs/newton_cartpole_sysid/result.json
```

Pendulum:

```bash
PYTHONPATH=. python scripts/pendulum/render_sysid_summary.py \
  outputs/newton_pendulum_sysid/result.json
```

Double pendulum:

```bash
PYTHONPATH=. python scripts/double_pendulum/render_sysid_summary.py \
  outputs/newton_double_pendulum_sysid/result.json
```

### 3. Batch sysid

Cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_batch_sysid.py \
  --fit-param init_pole_angle \
  --gt-value 0.20 \
  --init-value 0.65 \
  --env-count 8 \
  --iters 40 \
  --output-json outputs/newton_cartpole_batch_sysid/result.json
```

Pendulum:

```bash
PYTHONPATH=. python scripts/pendulum/newton_pendulum_batch_sysid.py \
  --fit-param init_angle \
  --gt-value 0.20 \
  --init-value 0.65 \
  --env-count 8 \
  --iters 40 \
  --output-json outputs/newton_pendulum_batch_sysid/result.json
```

Double pendulum:

```bash
PYTHONPATH=. python scripts/double_pendulum/newton_double_pendulum_batch_sysid.py \
  --fit-param init_angle_2 \
  --gt-value 0.10 \
  --init-value -0.40 \
  --env-count 8 \
  --iters 40 \
  --output-json outputs/newton_double_pendulum_batch_sysid/result.json
```

### 4. Batch summary and compare render

Each per-robot `render_batch_sysid_summary.py` wrapper generates the batch summary JSON/PNG and the stitched `gt | init | fit` compare render beside the selected batch `result.json`.

Cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/render_batch_sysid_summary.py \
  outputs/newton_cartpole_batch_sysid/result.json
```

Pendulum:

```bash
PYTHONPATH=. python scripts/pendulum/render_batch_sysid_summary.py \
  outputs/newton_pendulum_batch_sysid/result.json
```

Double pendulum:

```bash
PYTHONPATH=. python scripts/double_pendulum/render_batch_sysid_summary.py \
  outputs/newton_double_pendulum_batch_sysid/result.json
```

## Supported systems

| System | Single-run sysid | Batch sysid | Summary wrapper | Batch summary wrapper |
| --- | --- | --- | --- | --- |
| Cartpole | `scripts/cartpole/newton_cartpole_sysid.py` | `scripts/cartpole/newton_cartpole_batch_sysid.py` | `scripts/cartpole/render_sysid_summary.py` | `scripts/cartpole/render_batch_sysid_summary.py` |
| Pendulum | `scripts/pendulum/newton_pendulum_sysid.py` | `scripts/pendulum/newton_pendulum_batch_sysid.py` | `scripts/pendulum/render_sysid_summary.py` | `scripts/pendulum/render_batch_sysid_summary.py` |
| Double pendulum | `scripts/double_pendulum/newton_double_pendulum_sysid.py` | `scripts/double_pendulum/newton_double_pendulum_batch_sysid.py` | `scripts/double_pendulum/render_sysid_summary.py` | `scripts/double_pendulum/render_batch_sysid_summary.py` |

Current fit parameters:
- cartpole: `init_pole_angle`, `init_cart_pos`, `init_cart_vel`, `init_pole_angvel`, `cart_armature`, `cart_stiffness`, `cart_damping`, `pole_armature`, `pole_stiffness`, `pole_damping`
- pendulum: `init_angle`, `init_angvel`, `hinge_armature`, `hinge_stiffness`, `hinge_damping`
- double pendulum: `init_angle_1`, `init_angle_2`, `init_angvel_1`, `init_angvel_2`, `joint1_armature`, `joint1_stiffness`, `joint1_damping`, `joint2_armature`, `joint2_stiffness`, `joint2_damping`

## Outputs

Single-run and batch workflows both write artifacts next to the selected `result.json`.

Expected artifacts:
- `result.json`
- `summary.json`
- `summary.png`
- `compare/...gif`
- `compare/...mp4`

Typical layout:

```text
outputs/newton_cartpole_sysid/
├── result.json
├── summary.json
├── summary.png
└── compare/
    ├── gt/
    ├── init/
    ├── fit/
    └── compare/
        ├── cartpole_compare.gif
        └── cartpole_compare.mp4
```

Batch runs use the same layout pattern under:
- `outputs/newton_cartpole_batch_sysid/`
- `outputs/newton_pendulum_batch_sysid/`
- `outputs/newton_double_pendulum_batch_sysid/`

## Lower-level render helpers

The preferred entrypoints are the per-robot summary wrappers above.

Lower-level helpers still exist when you want finer control:
- `scripts/cartpole/render_newton_cartpole.py`
- `scripts/pendulum/render_newton_pendulum.py`
- `scripts/double_pendulum/render_newton_double_pendulum.py`
- `scripts/cartpole/render_compare_cartpole.py`
- `scripts/pendulum/render_compare_pendulum.py`
- `scripts/double_pendulum/render_compare_double_pendulum.py`

## Troubleshooting

If `warp` or `newton` cannot be imported:

```bash
PYTHONPATH=. python -c "import warp as wp, newton; print('Warp:', wp.__version__)"
```

If local imports fail:
- run from the repo root
- keep `PYTHONPATH=.`
- use the repo-local `.venv`

If rendering works but GIF/MP4 generation fails, verify `ffmpeg` is installed and available on `PATH`.

## Notes

- Angle handling may depend on the script's `--angle-mode` option when applicable.
- The current workflows fit one parameter at a time against trajectory loss, including both initial-state and selected joint dynamics parameters.
- Summary wrappers are the main user-facing interface for generating plots and comparison renders from a finished result.
