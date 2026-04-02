# diffsysid

Differentiable system identification with NVIDIA Newton + Warp.

This repo currently focuses on three URDF-backed systems:
- cartpole
- pendulum
- double pendulum

Each system supports:
- saved ground-truth trajectory generation
- single-run sysid
- multi-parameter single-run sysid
- true world-batched GPU batch sysid with multi-start restarts
- multi-parameter batch sysid
- initial-state and dynamics-parameter fitting
- summary generation
- side-by-side `gt | init | fit` Newton renders

## Project structure

```text
diffsysid/
├── diffsysid/                         # Shared optimization and batch utilities
├── data/
│   ├── urdf/
│   │   ├── cartpole.urdf
│   │   ├── pendulum.urdf
│   │   └── double_pendulum.urdf
│   └── gt_datasets/                  # Saved ground-truth rollout JSON files
├── scripts/
│   ├── batch_sysid_common.py         # Shared Newton sysid helpers
│   ├── gt_trajectory_common.py       # Shared saved-ground-truth helpers
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

### 1. Generate Ground-Truth Trajectory

Generate a ground-truth rollout once and save it to `data/gt_datasets/`:

Cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/generate_cartpole_gt.py \
  --fit-param pole_damping \
  --gt-value 0.50 \
  --steps 120 \
  --output-json data/gt_datasets/cartpole_pole_damping_gt.json
```

Pendulum:

```bash
PYTHONPATH=. python scripts/pendulum/generate_pendulum_gt.py \
  --fit-param hinge_stiffness \
  --gt-value 8.0 \
  --steps 120 \
  --output-json data/gt_datasets/pendulum_hinge_stiffness_gt.json
```

Double pendulum:

```bash
PYTHONPATH=. python scripts/double_pendulum/generate_double_pendulum_gt.py \
  --fit-param joint2_stiffness \
  --gt-value 8.0 \
  --steps 120 \
  --output-json data/gt_datasets/double_pendulum_joint2_stiffness_gt.json
```

Long-horizon cartpole example:

```bash
PYTHONPATH=. python scripts/cartpole/generate_cartpole_gt.py \
  --fit-param pole_damping \
  --gt-value 0.50 \
  --steps 720 \
  --output-json data/gt_datasets/cartpole_pole_damping_gt_long.json
```

Generator convenience options:
- `--fit-params all` expands to every supported parameter for that robot
- `--fit-params all_dynamics` expands to only dynamics parameters
- `--gt-value random` or `--gt-values random` samples ground-truth values around the generator defaults
- `--random-gt-span` controls that sampling range

Multi-parameter upright cartpole example:

```bash
PYTHONPATH=. python scripts/cartpole/generate_cartpole_gt.py \
  --fit-params init_pole_angle pole_damping \
  --gt-values 0.0 0.5 \
  --angle-mode top_offset \
  --steps 120 \
  --output-json data/gt_datasets/cartpole_multi_gt.json
```

Fit-all random cartpole GT example:

```bash
PYTHONPATH=. python scripts/cartpole/generate_cartpole_gt.py \
  --fit-params all \
  --gt-values random \
  --angle-mode top_offset \
  --steps 120 \
  --output-json data/gt_datasets/cartpole_multi_gt.json
```

### 2. Single-run sysid

The single-run scripts always fit against a saved trajectory loaded from `--gt-json`.

They support both:
- scalar mode with `--fit-param`, `--init-value`
- vector mode with `--fit-params`, `--init-values`

You can fit either initial-state parameters or dynamics parameters, but the target rollout must be generated first.

Convenience options:
- `--fit-params all` expands to every supported parameter for that robot
- `--fit-params all_dynamics` expands to only dynamics parameters
- `--init-value random` or `--init-values random` samples the initial guess from a uniform range around the saved ground-truth values
- `--random-init-span` controls that sampling range

Cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_sysid.py \
  --gt-json data/gt_datasets/cartpole_pole_damping_gt.json \
  --fit-param pole_damping \
  --init-value 1.00 \
  --iters 80 \
  --output-json outputs/newton_cartpole_sysid/result.json
```

Pendulum:

```bash
PYTHONPATH=. python scripts/pendulum/newton_pendulum_sysid.py \
  --gt-json data/gt_datasets/pendulum_hinge_stiffness_gt.json \
  --fit-param hinge_stiffness \
  --init-value 8.5 \
  --iters 80 \
  --output-json outputs/newton_pendulum_sysid/result.json
```

Double pendulum:

```bash
PYTHONPATH=. python scripts/double_pendulum/newton_double_pendulum_sysid.py \
  --gt-json data/gt_datasets/double_pendulum_joint2_stiffness_gt.json \
  --fit-param joint2_stiffness \
  --init-value 8.5 \
  --iters 80 \
  --output-json outputs/newton_double_pendulum_sysid/result.json
```

Multi-parameter example, cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_sysid.py \
  --gt-json data/gt_datasets/cartpole_pole_damping_gt.json \
  --fit-params init_pole_angle pole_damping \
  --init-values 0.60 1.00 \
  --iters 40 \
  --output-json outputs/newton_cartpole_sysid/multi_param_run.json
```

Fit-all random-init example, cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_sysid.py \
  --gt-json data/gt_datasets/cartpole_pole_damping_gt.json \
  --fit-params all \
  --init-values random \
  --random-init-span 0.1 \
  --seed 7 \
  --iters 40 \
  --output-json outputs/newton_cartpole_sysid/all_random_run.json
```

Multi-parameter example, pendulum:

```bash
PYTHONPATH=. python scripts/pendulum/newton_pendulum_sysid.py \
  --gt-json data/gt_datasets/pendulum_hinge_stiffness_gt.json \
  --fit-params init_angle hinge_stiffness \
  --init-values 0.40 8.5 \
  --iters 40 \
  --output-json outputs/newton_pendulum_sysid/multi_param_run.json
```

Multi-parameter example, double pendulum:

```bash
PYTHONPATH=. python scripts/double_pendulum/newton_double_pendulum_sysid.py \
  --gt-json data/gt_datasets/double_pendulum_joint2_stiffness_gt.json \
  --fit-params init_angle_2 joint2_stiffness \
  --init-values 0.30 8.5 \
  --iters 40 \
  --output-json outputs/newton_double_pendulum_sysid/multi_param_run.json
```

Long-horizon saved-ground-truth example, cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_sysid.py \
  --gt-json data/gt_datasets/cartpole_pole_damping_gt_long.json \
  --fit-param pole_damping \
  --init-value 1.00 \
  --iters 40 \
  --output-json outputs/newton_cartpole_sysid/from_long_gt.json
```

### 3. Single-run summary and compare render

Each per-robot `render_sysid_summary.py` wrapper does both:
- generates `summary.json`
- generates `summary.png`
- renders `gt`, `init`, and `fit`
- stitches them into a compare GIF and MP4 under `compare/`

The selected result JSON is a positional argument. Summaries and renders are written beside that file.

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

### 4. Batch sysid

The batch scripts also always fit against a saved trajectory loaded from `--gt-json`.

They support both scalar and multi-parameter fitting. In batch mode, each environment carries its own candidate parameter vector and elite restart operates on that full vector.

Batch execution is now true Newton world batching:
- one replicated Newton model contains `env-count` worlds
- all candidate environments are stepped together in one batched GPU pass per iteration
- snippet mode replicates `snippet_count * env_count` worlds and aggregates losses back to each candidate
- this is different from the older Python-loop batch mode where candidates were evaluated one-by-one

Convenience options:
- `--fit-params all` expands to every supported parameter for that robot
- `--fit-params all_dynamics` expands to only dynamics parameters
- `--init-value random` or `--init-values random` samples the population center from a uniform range around the saved ground-truth values
- `--random-init-span` controls that sampling range before the per-env `--init-span` jitter is applied

For longer ground-truth trajectories, batch sysid also supports snippet mode:
- keep `--steps` as the full ground-truth horizon
- set either `--snippet-duration` or `--snippet-steps` to break that long horizon into shorter windows
- each optimization candidate is then evaluated on every snippet, so total rollout instances become `snippet_count * env_count`
- multi-snippet mode currently supports dynamics-parameter fitting only, because each snippet starts from the ground-truth state at that window boundary

Saved ground-truth JSON files are expected to live under `data/gt_datasets/`.

Cartpole:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_batch_sysid.py \
  --gt-json data/gt_datasets/cartpole_pole_damping_gt.json \
  --fit-param pole_damping \
  --init-value 1.00 \
  --env-count 8 \
  --iters 40 \
  --output-json outputs/newton_cartpole_batch_sysid/result.json
```

Pendulum:

```bash
PYTHONPATH=. python scripts/pendulum/newton_pendulum_batch_sysid.py \
  --gt-json data/gt_datasets/pendulum_hinge_stiffness_gt.json \
  --fit-param hinge_stiffness \
  --init-value 8.5 \
  --env-count 8 \
  --iters 40 \
  --output-json outputs/newton_pendulum_batch_sysid/result.json
```

Double pendulum:

```bash
PYTHONPATH=. python scripts/double_pendulum/newton_double_pendulum_batch_sysid.py \
  --gt-json data/gt_datasets/double_pendulum_joint2_stiffness_gt.json \
  --fit-param joint2_stiffness \
  --init-value 8.5 \
  --env-count 8 \
  --iters 40 \
  --output-json outputs/newton_double_pendulum_batch_sysid/result.json
```

Multi-parameter example, cartpole batch:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_batch_sysid.py \
  --gt-json data/gt_datasets/cartpole_pole_damping_gt.json \
  --fit-params init_pole_angle pole_damping \
  --init-values 0.60 1.00 \
  --init-span 0.2 \
  --env-count 8 \
  --iters 20 \
  --output-json outputs/newton_cartpole_batch_sysid/multi_param_run.json
```

Fit-all random-init example, cartpole batch:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_batch_sysid.py \
  --gt-json data/gt_datasets/cartpole_pole_damping_gt.json \
  --fit-params all \
  --init-values random \
  --random-init-span 0.1 \
  --seed 7 \
  --env-count 8 \
  --iters 20 \
  --output-json outputs/newton_cartpole_batch_sysid/all_random_run.json
```

Multi-parameter example, pendulum batch:

```bash
PYTHONPATH=. python scripts/pendulum/newton_pendulum_batch_sysid.py \
  --gt-json data/gt_datasets/pendulum_hinge_stiffness_gt.json \
  --fit-params init_angle hinge_stiffness \
  --init-values 0.40 8.5 \
  --init-span 0.3 \
  --env-count 8 \
  --iters 20 \
  --output-json outputs/newton_pendulum_batch_sysid/multi_param_run.json
```

Multi-parameter example, double pendulum batch:

```bash
PYTHONPATH=. python scripts/double_pendulum/newton_double_pendulum_batch_sysid.py \
  --gt-json data/gt_datasets/double_pendulum_joint2_stiffness_gt.json \
  --fit-params init_angle_2 joint2_stiffness \
  --init-values 0.30 8.5 \
  --init-span 0.3 \
  --env-count 8 \
  --iters 20 \
  --output-json outputs/newton_double_pendulum_batch_sysid/multi_param_run.json
```

Long-horizon snippet example, cartpole batch:

```bash
PYTHONPATH=. python scripts/cartpole/newton_cartpole_batch_sysid.py \
  --gt-json data/gt_datasets/cartpole_pole_damping_gt_long.json \
  --fit-params all_dynamics \
  --init-value random \
  --env-count 16 \
  --snippet-duration 0.5 \
  --iters 10 \
  --output-json outputs/newton_cartpole_batch_sysid/from_long_gt.json
```

With the default `dt = 1/240`, that example uses the saved 3-second ground-truth rollout, splits it into six 0.5-second snippets, and evaluates `6 * 16 = 96` rollout instances per optimization iteration.

Saved-ground-truth example, pendulum batch:

```bash
PYTHONPATH=. python scripts/pendulum/newton_pendulum_batch_sysid.py \
  --gt-json data/gt_datasets/pendulum_hinge_stiffness_gt.json \
  --fit-param hinge_stiffness \
  --init-value 8.5 \
  --env-count 8 \
  --iters 20 \
  --output-json outputs/newton_pendulum_batch_sysid/from_gt.json
```

### 5. Batch summary and compare render

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

Dynamics-only aliases:
- cartpole: `cart_armature`, `cart_stiffness`, `cart_damping`, `pole_armature`, `pole_stiffness`, `pole_damping`
- pendulum: `hinge_armature`, `hinge_stiffness`, `hinge_damping`
- double pendulum: `joint1_armature`, `joint1_stiffness`, `joint1_damping`, `joint2_armature`, `joint2_stiffness`, `joint2_damping`

## Outputs

Single-run and batch workflows both write artifacts next to the selected `result.json`.

The summary wrappers accept any result JSON path, for example:
- `outputs/newton_cartpole_sysid/result.json`
- `outputs/newton_cartpole_sysid/multi_param_run.json`
- `outputs/newton_cartpole_batch_sysid/multi_param_run.json`

Expected artifacts:
- `result.json`
- `summary.json`
- `summary.png`
- `compare/...gif`
- `compare/...mp4`

Snippet-enabled batch runs also record a `snippet_batching` section in the result JSON with:
- snippet count
- snippet step/time ranges
- per-snippet env count
- total rollout instance count

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

Current behavior:
- summary wrappers always write `summary.json` and `summary.png` to the result file's parent directory
- running a summary wrapper on a different result file in the same folder will overwrite the previous summary artifacts
- the underlying result JSON files are not overwritten unless you reuse the same `--output-json`

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
- The current workflows support both one-parameter and multi-parameter fitting against trajectory loss.
- Multi-parameter results store `fit_params` plus per-parameter values and gradients in the JSON output.
- Summary wrappers are the main user-facing interface for generating plots and comparison renders from a finished result.
- Batch sysid uses replicated Newton worlds so `env-count` candidates run in one batched GPU simulation rather than a Python loop of separate solves.
