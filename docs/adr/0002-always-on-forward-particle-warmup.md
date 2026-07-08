# ADR-0002: Always-on forward particle warmup (hidden T_w = 100)

**Status:** accepted
**Date:** 2026-07-08

## Context

Uniform-random particle positions are not samples from the particle phase-space stationary distribution — inertial particles preferentially concentrate, so un-warmed initial distributions bias the DA experiment relative to what a physical observation of a developed flow would look like. A warmup existed only as an opt-in velocity-init mode for the Stokes-number study (`Warmup_Vel_Init`), and it ran **backward**: it grabbed the snapshot `T_w` *before* the target IC, which required `T_w` to be a multiple of `t_skip`, required the snapshot index to be ≥ `T_w/t_skip`, and was blocked for tracers.

## Decision

Warmup is **always on**, runs **forward**, and applies to tracers and inertial particles alike. The warmup time is a **hidden constant, `T_w = 100` time units** — deliberately not exposed in the YAML, so every experiment uses the same value.

Mechanics, per case (with `idx` = the case's permuted snapshot index, `k = T_w / t_skip`):

- The **true IC is always the stored snapshot `idx + k`** — never a solver-integrated state — so it is bit-identical across particle-init variants. The without-replacement permutation (ADR-0001) draws `idx` from `[0, N−1−k]` so `idx + k` always exists.
- **Equilibrium seeding** (the default, any St): scatter particles uniform-random on `snapshot[idx]`, co-evolve flow + particles with the solver for `T_w`, and keep only the final **particle** state as the particle IC. The integrated flow state is discarded in favor of the stored snapshot. Tracers run the same co-evolution for pipeline consistency, even though uniform is already their stationary spatial distribution.
- **Gaussian seeding** (option, requires St > 0): uniform-random positions plus i.i.d. N(0, std²) velocities placed directly at t = 0 — no particle warmup, no solver run. The true IC is the same `snapshot[idx + k]`, so equilibrium-vs-Gaussian comparisons hold the flow fixed.

Config surface: `particle_init: equilibrium | gaussian` (optional key, default `equilibrium`), with the Gaussian std as its existing sub-key. The old `fluid` / `gaussian` / `warmup` trio and `Fluid_Vel_Init` are deleted.

## Consequences

- Equilibrium seeding costs one co-evolution of `T_w/dt` solver steps per (case, n_particles) combination (~10⁴ steps at dt = 10⁻²), including for tracers where it is statistically a no-op — accepted for uniformity.
- `T_w` must be a multiple of `t_skip`; with `T_w` hidden at 100 this is a startup assertion, not a user knob.
- Changing `T_w` is a code change and would invalidate comparability with prior results; if it ever needs to vary, supersede this ADR rather than quietly adding a config key.
