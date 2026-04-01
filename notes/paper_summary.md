# Paper summary: Few-Shot Neural Differentiable Simulator

## One-line summary
Use a tiny set of real cube-collision trajectories to calibrate an analytical simulator (MuJoCo), scale synthetic contact-rich data from it, and train a differentiable mesh/GNN rigid-contact simulator that matches real trajectories better than differentiable analytical baselines.

## Core pipeline
1. Collect a few real trajectories with object 6D poses.
2. Identify MuJoCo contact parameters with CMA-ES.
3. Generate a much larger synthetic dataset from calibrated MuJoCo.
4. Train a FIGNet-style mesh GNN on that synthetic data.
5. Add surrogate gradients for collision-detection outputs.
6. Use the learned simulator for rollout prediction and gradient-based optimization.

## Concrete experimental setup from the paper
- Objects: 3D-printed cubes
- Scenario: one cube pushes/collides with another on a tabletop
- Sensing: AprilTags + TagSLAM
- Cameras: 4 × Intel RealSense D435if @ 60 Hz
- Original real training data: 3 trajectories, ~20 frames each
- Real test data: 14 trajectories
- Scaled synthetic training data: 3000 trajectories
- Extra qualitative scene: one cube hitting a bowling-like set of 10 cubes

## Important reproduction uncertainties
- Exact network depth/width and training hyperparameters are not fully specified in the text extracted from the PDF.
- Exact data format and MuJoCo XML/model details are not included.
- The collision library integration details (Coal + GJK/EPA) may require code release or engineering interpretation.

## Current local reproduction status
- Preferred paper path is now `/home/yu-tang-liu/Downloads/paper/2603.06218_Few-Shot_Neural_Differentiable_Simulator_Real-to-Sim_Rigid-Contact_Modeling.pdf`.
- `/home/yu-tang-liu/.venv` already has `mujoco==3.5.0`, so the identification stage can start without new installs.
- `scripts/mujoco_sysid_scaffold.py` now provides a minimal 2-cube MuJoCo identification baseline using synthetic observations and a simple random-search fitter.
- Missing for closer paper reproduction: `torch`, `optuna`/CMA-ES, graph/mesh pipeline, and real AprilTag/TagSLAM trajectories.
