# ADR-0007: Observer track plot with measurement resets

**Status:** accepted
**Date:** 2026-07-23

## Context

`plot_particle_tracks` (`case_post_proc.py:135-230`) draws the **reconstructed trajectory**'s particle component as one continuous dashed line per particle, forward-simulated once from the optimized IC over the whole window (`DA_trj = trj_gen_fn(omega0_DA_hat, *particle_IC)`, `DA_engine.py:355`).

That line does not reflect how the loss is actually computed. Inside `create_loss_fn`'s `lax.scan` (`loss_funcs.py:40-143`), particle position — and, when `optimize_velocity` is set, particle velocity — is overwritten with the optimizer's per-measurement estimate at every masked measurement index (`have_measurement`, `loss_funcs.py:46-118`), then integrated forward only until the next measurement. The vorticity state is never reset; it stays continuous through the whole window. This reset mechanism exists only inside the loss computation and its trajectory is discarded — nothing plots or saves it today.

The plot should show this reset behavior instead of the misleadingly smooth continuous rollout, since the continuous line overstates how far the optimizer's particle-position estimate is allowed to drift between measurements.

## Decision

**New terms** (added to `CONTEXT.md`; see below) — "observer track", "measurement reset", "reset value", "drift point" — describe this per-measurement mechanism, kept distinct from **cycled DA**'s "segment" (a full observation window, not a sub-window reset).

1. **`create_loss_fn` gains a `return_traj` flag.** When set, the scan's `body` also carries/collects the full particle state (`xp_DA, yp_DA`, and `up_DA, vp_DA` for inertial) as `lax.scan` auxiliary output, alongside the existing per-step loss. Two values are captured at each masked step: the **drift point** (state entering the step, i.e. the free-running value just before the reset is applied) and the state after the reset+integrate. This guarantees the plotted track can never drift out of sync with the actual optimization logic, since both use the same code path. The existing gradient-computation call site (`return_traj` defaulting False) is unaffected.
2. **Reset value follows the same branch the loss already uses**: noisy measurement (`xp_meas_traj`/`yp_meas_traj`, plus `up_meas_traj`/`vp_meas_traj` when available) in the plain tracer path; `PP_opt` (position **and** velocity) in the inertial/`Joint_Opt` path. No new "ground truth" substitution — the plot shows exactly what the optimizer reset to.
3. **`_run_DA_case`/`post_proc_case_main` compute the observer track once**, after optimization, via `create_loss_fn(..., return_traj=True)` fed with the final `omega0_DA_hat` (and `PP_opt` where applicable), and save it to new `.npy` files (e.g. `xp_DA_obs.npy`, `yp_DA_obs.npy`, drift-point arrays, plus velocity equivalents for inertial cases) — **separate from** the existing `xp_DA.npy`/`yp_DA.npy`, which keep saving the unchanged continuous **reconstructed trajectory**. Likewise, `omega_DA_trj.npy` and the vorticity-based reconstruction-error metrics in `results.parquet` are untouched — this change is scoped to the particle-position/velocity track only.
4. **`plot_particle_tracks` is updated**: the dashed DA line is replaced by the observer track (drawn as a sequence of free-running segments), with a thin dotted connector at each measurement linking the **drift point** to the **reset value** — the connector length is the per-measurement correction magnitude, deliberately visible rather than smoothed away. The **target trajectory** line/markers are unchanged. The existing hollow "DA at measurement" marker is repurposed to sit at the drift point (the post-reset position now coincides with the reset value, which would otherwise render redundantly on top of the target/measurement marker).
5. **`replot_case.py` is updated** to load the new observer-track `.npy` files (instead of `xp_DA.npy`/`yp_DA.npy`) when calling `plot_particle_tracks`, so it keeps regenerating this plot from saved arrays alone, with no access to `create_loss_fn`/optimization internals required.

## Consequences

- The particle-tracks plot now shows two materially different things depending on path: for `plot_vort_comp` and reconstruction-error metrics, "DA" still means the fully continuous reconstructed trajectory (`omega_DA_trj.npy`, `xp_DA.npy`/`yp_DA.npy`); for the tracks plot specifically, "DA"/"observer" means the reset-corrected track (`xp_DA_obs.npy`/`yp_DA_obs.npy`). Anyone extending `case_post_proc.py` needs to know which one a given diagnostic should use.
- A few extra small `.npy` files are written per case (observer-track positions/velocities/drift points); negligible relative to existing per-case output.
- `replot_case.py` gains a dependency on these new files for cases post-processed before this change — old cases lacking `xp_DA_obs.npy` cannot regenerate the new-style tracks plot (only the old continuous one), unless reprocessed.
