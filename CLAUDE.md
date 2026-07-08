# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Kolmogorov Flow Data Assimilation (KF_DA)** — a research project combining 2D pseudo-spectral fluid simulation, Lagrangian particle tracking, and inverse-problem optimization. The scientific goal is to infer a flow's initial condition from noisy particle position observations.

Key physical model: forced 2D Navier-Stokes (Kolmogorov flow) on a periodic domain, solved with a pseudo-spectral method + RK4 time stepping.

## Environment & Build

This project uses **UV** as the package manager. Python 3.14 is required (see `.python-version`).

```bash
uv venv && source .venv/bin/activate
uv sync --frozen          # Install locked dependencies
```

Docker builds are published to GHCR via `.github/workflows/publish-ghcr.yaml`:
```bash
docker build -t kf-da:latest .
docker run --gpus all kf-da:latest python main_scripts/DA_exp_ctrl.py
```

## Running Experiments

Configuration is loaded from an external YAML file at `../kf-da-configs/daExpConfig.yaml` (outside this repo). The main entry points are:

```bash
python main_scripts/DA_exp_ctrl.py        # Main DA experiment loop
python main_scripts/trj_generator.py      # Trajectory generation / animation
```

Diagnostic scripts in `diag_scripts/` are standalone analyses (Lyapunov exponents, particle entropy, etc.) and can be run directly.

## Architecture

### Core Data Flow

```
DA_exp_ctrl.py
  └─ DA_engine.py::DA_exp_main()
       ├─ Loads attractor snapshots (pre-computed)
       ├─ Master permutation (seed 0) assigns each case a unique true-IC snapshot
       ├─ For each case seed → true IC (snapshot idx + T_w offset, ADR-0002)
       │    → warmed particle IC → target trajectory + observations
       └─ For each (T, n_particles, NT, optimizer, loss crit):
            ├─ Initial guess from attractor (velInit/, fold_in(seed, 0))
            ├─ Parametrize IC via Fourier modes (icParam/Fourier_Param.py)
            └─ Optimize IC to match particle observations
                 ├─ Loss: MSE on particle positions (loss_funcs.py)
                 ├─ Gradients: adjoint solver (adjoint.py)
                 └─ Optimizer: BFGS / Quasi-Newton (opti/)
```

Seeding follows ADR-0001: one integer seed per DA case (`n_cases` in the YAML) drives the IC guess, particle positions, and observation noise through independent `fold_in` substreams; true ICs are drawn without replacement via a master permutation. Particle initialization follows ADR-0002: always-on forward warmup of T_w = 100 time units (hidden constant), `particle_init: equilibrium | gaussian`.

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/kf_da/solver/solver.py` | Core physics: `Forced_2D_NS`, `KF_Stepper`, `KF_TP_Stepper` (flow + particles), `Omega_Integrator` |
| `src/kf_da/daComp/DA_engine.py` | Experiment orchestration; outer loop over case seeds, inner loops over parameters and optimizers |
| `src/kf_da/daComp/configs.py` | Config dataclasses: `KF_Opts`, `DA_Opts` (incl. `sigma_vy`/`vx__vy_sigma` for inertial velocity noise), `Particle_Opts` |
| `src/kf_da/daComp/adjoint.py` | Reverse-mode AD for gradients and Hessian-vector products |
| `src/kf_da/daComp/loss_funcs.py` | `MSE_PP` (particle positions), `MSE_Vel` (velocity); measurement masks `t_mask`; `optimize_velocity` flag extends `PP_opt` to include inertial particle velocities |
| `src/kf_da/opti/optimization.py` | `BFGS`, `Joint_Opt` (tracer path: `set_pp_loss_fn`; inertial path: `set_inertial_pp_loss_fn`), `Loss_and_Deriv_fns`; `ArmijoLineSearch` in `LS_TR.py` |
| `src/kf_da/icParam/Fourier_Param.py` | Fourier mode parametrization; optional spectral preconditioning via `precond_beta` (default off, ADR-0004) |
| `src/kf_da/velInit/AI.py` | Attractor-based IC initialization (random sample from attractor snapshots) |

### JAX Usage

The solver and adjoint code are JAX-first (JIT-compiled, GPU-capable). Computations operate on spectral (Fourier) coefficients of vorticity. `jax.grad` / `jax.vjp` underpin the adjoint solver. All computation is double precision (`jax_enable_x64` set at entry, ADR-0003).

### Results Storage

Results are written as **Parquet files** (via pandas + PyArrow/fastparquet), organized hierarchically by case seed, Re, and experiment parameters (`case=N/T=…/np=…/NT=…/{crit}/{opt}/{param}/`). `case_post_proc.py` and `gPost/global_post_main.py` handle post-processing and Excel export.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (`nfrank22-gh/KF_DA`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
