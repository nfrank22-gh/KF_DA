# ADR-0001: One seed per DA case

**Status:** accepted
**Date:** 2026-07-08

## Context

Experiment replication was structured as a cross product of three independent seed axes: `TIC_seed_list` (true IC selection), `PIC_seed_list` (particle initial positions, in practice always `[0]`), and `num_opt_inits` (IC guess draws), nested inside the parameter sweeps. This made "how many independent experiments did we run" hard to state, allowed accidental correlation (every true IC shared the same particle configuration), and required resume hacks (`count_folders` offsetting of guess indices).

Independent per-seed draws of the true IC also permitted two replicates to silently pick the same attractor snapshot.

## Decision

A **DA case** is one integer seed. Config specifies `n_cases`; case `i` uses seed `i` (0..n_cases−1). The old keys `num_Tic`, `num_opt_inits`, and `PIC_seed_list` are removed.

Randomness per case:

- **True IC**: drawn **without replacement** — a master RNG (seed hardcoded to 0) permutes the eligible attractor snapshot indices once; case `i` takes `permutation[i]`. Distinct cases are guaranteed distinct true ICs, and case `i`'s true IC is stable when `n_cases` is later increased.
- **IC guess**: `jax.random.fold_in(seed, 0)` drives attractor initialization.
- **Particle initial positions**: `fold_in(seed, 1)`.
- **Observation noise**: position noise from `fold_in(seed, 2)`, velocity noise from `fold_in(seed, 3)` (replacing the old `PRNGKey(PIC_seed)` derivation).

The parameter sweeps (`T_list`, `n_particles_list`, `NT_list`, optimizer, loss) still cross against every case.

Results are a **clean break**: directory tree becomes `root/case=N/T=…/np=…/NT=…/{crit}/{opt}/{param}/` (the `PI/seed_M`, `cases/K`, and float-precision levels disappear); the parquet schema replaces `true_IC_seed` / `PIC_seed` / `init_IC_seed` with a single `case_seed` column. Resume = skip a case leaf directory that already exists; adding replicates = raising `n_cases`.

## Consequences

- Old result trees remain on disk but are not readable by the new post-processing; rerun to regenerate.
- `n_cases` may not exceed the number of eligible attractor snapshots (the permutation errors out otherwise).
- The hierarchical case summary must be rewritten against the new single-seed schema.
- Holding the flow fixed while varying particle configurations is no longer expressible as a config sweep; if that experiment returns, it needs a deliberate new mechanism (do not resurrect the seed cross product silently — supersede this ADR).
