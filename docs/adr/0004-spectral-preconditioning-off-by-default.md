# ADR-0004: Spectral preconditioning off by default

**Status:** accepted
**Date:** 2026-07-08

## Context

The Fourier parametrization applies spectral preconditioning P(k) = exp(−½ν β k²), producing the damped representation the optimizer searches in. β was hardcoded to 0.1 in `DA_exp_ctrl.py`, so every experiment ran preconditioned with no way to switch it off from config. β = 0 makes P the identity exactly, so "no preconditioning" needs no separate code path.

## Decision

β becomes an optional YAML key: `daSet.precond_beta`, read as `da_set.get("precond_beta", 0.0)` and passed straight into `Fourier_Param`. Key absent → β = 0 → preconditioning off. Setting `precond_beta: 0.1` restores the previous behavior. `Fourier_Param` itself is unchanged; its `__repr__` already embeds β, so preconditioned and unpreconditioned runs land in distinct result directories.

## Consequences

- Default experiments optimize directly in (truncated, packed) Fourier-mode space; the **damped representation** is only meaningfully "damped" when `precond_beta > 0`.
- Results produced before this change all used β = 0.1 — compare against new defaults with care.
- The identity transform still performs the exp(0)-scaling multiplies; accepted as negligible.
