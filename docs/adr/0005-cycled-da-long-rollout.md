# ADR-0005: Cycled (segmented) DA over a long time horizon

**Status:** accepted
**Date:** 2026-07-20

## Context

The main engine (`DA_engine.py`) assimilates a single observation window `T`
(one Lyapunov time, `T_dict[Re]`) and reports how close the recovered IC is to
truth. It does not explore what happens when DA is chained over many consecutive
windows — i.e. whether the estimate stays locked to the reference over a long
horizon or drifts. We want a dedicated script for this **cycled 4D-Var** study,
kept deliberately simple: **tracer particles only (St = 0), no observation
noise**, so the only moving part is the segment-to-segment handoff.

## Decision

A new `main_scripts` entry point runs a chain of DA optimizations over
`n_segments` consecutive windows, each of length `T_seg = T_dict[Re]` (the
Lyapunov timescale, exactly as the main experiments — **not** a config knob).
Total horizon = `T_dict[Re] * n_segments`.

### Cycling mechanics (the handoff)

Each segment is an assimilation window whose analysis is at the window *start*:

- **Segment 1** initial guess = `AI(min_norm, max_norm)` draw from the attractor,
  a controlled distance from truth (same mechanism as the main engine).
- **Segment k → k+1 handoff:** optimize segment `k`'s IC (`omega0_DA_k` at
  `t_k`), roll that analysis forward through the window to `t_{k+1}`, and use
  that **forecast state as segment k+1's initial guess (background)**. This is
  standard cycled 4D-Var. The forecast, not the raw optimized IC, transfers.

### Reference & observations

- **True IC** = one attractor snapshot selected by the config `seed`; the
  reference is one continuous rollout over the whole horizon.
- **Particles** are seeded **once** at `t = 0` via `Equilibrium_Init` (ADR-0002)
  and advected **continuously** over the entire horizon under the reference
  flow. Each segment observes the sub-slice of positions in its window; the
  segment's DA particle IC = reference particle positions at `t_k`.
- **Observations** = particle positions at `NT` evenly spaced times per window
  (`get_tmask`, `m_dt = None`). `sigma_y = 0` (no noise); loss = `MSE_PP`;
  optimizer = `BFGS` (tracer path — no `Joint_Opt`, no velocity optimization).

### Outputs

The full-horizon **DA trajectory** is the concatenation of each segment's
analysis rollout (each segment's optimized IC rolled forward through its own
window). Discontinuities at segment boundaries are the re-assimilation
corrections and are preserved, not smoothed.

Three figures, all regenerable from saved data:
1. **error-vs-time** over the whole horizon — relative spectral L2,
   `||w_DA_hat(t) - w_ref_hat(t)|| / attractor_rad` (matches
   `case_post_proc.rel_error`).
2. **side-by-side vorticity video** — reference | DA, synced over the horizon.
3. **loss-convergence plot** — per-segment optimizer loss curves stacked.

### Storage & structure (flat, no case/T/np tree)

Single flat directory `create_results_dir()/LongRollout/<descriptive-name>/`
holding **everything at every solver step**: full reference and DA vorticity
trajectories, particle trajectories, per-segment optimized ICs, per-segment
loss/grad-norm records, the error-vs-time array, and a config snapshot.

### Run modes & config surface

- YAML key `mode: run | post | run_post` — `run` does the cycled DA and saves
  data only; `post` loads saved data and renders figures/video; `run_post` both.
- YAML exposes physics + experiment only: `Re, NDOF, dt, n_segments,
  n_particles, NT, seed, min_norm, max_norm, mode`. Optimizer internals
  (`BFGS its=150`, line search), `Fourier K = NDOF//2`, and `St = 0` are
  hardcoded in the script, matching `DA_exp_ctrl.py`'s style.

## Consequences

- Saving full fields every step is ~GBs for long horizons (NDOF=128, dt=0.01:
  ~40 MB per Lyapunov window per trajectory). Accepted for zero-loss
  regeneration; the flat layout keeps it simple.
- The first segment's guess (controlled distance from truth) is structurally
  different from later segments' organic forecast backgrounds — expected and
  intrinsic to cycling.
- Continuous particles can thin coverage under chaotic stretching over very long
  horizons; if that becomes limiting, re-seeding per segment would supersede
  this choice.
- Reuses `Loss_and_Deriv_fns`, `BFGS`, `Fourier_Param`, `MSE_PP`,
  `Equilibrium_Init`, `AI`, and the solver trajectory generators unchanged.
