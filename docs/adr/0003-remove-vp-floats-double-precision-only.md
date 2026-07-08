# ADR-0003: Remove VP floats — double precision only

**Status:** accepted
**Date:** 2026-07-08

## Context

The variable-precision float study (custom C++ Pybind11 float format used to quantify how many mantissa bits the stored forward trajectory needs) concluded. Its machinery remained threaded through the whole pipeline: a `vp_list` axis in `DA_Opts`, a `vfloat` parameter in `Loss_and_Deriv_fns`, dedicated `*_vp_*` code paths in the adjoint solver, a float-precision level in the results directory tree, a `floatp` parquet column grouped over in global post-processing, a platform-gated prebuilt wheel in `pyproject.toml`/Dockerfile, and standalone diag scripts. Everything runs with `jax_enable_x64(True)`; the VP list was the only remaining "data type option."

## Decision

Full excision. Delete:

- `src/kf_da/vp_floats/` and `wheels/`, plus the `vpfloat` dependency in `pyproject.toml` (re-lock with `uv lock`) and its Dockerfile handling.
- The `*_vp_*` functions in `adjoint.py` and the `vfloat` parameter threaded through `Loss_and_Deriv_fns`.
- `DA_Opts.vp_list`, `VP_Float_Settings`, the float-precision results-directory level, and the `floatp` column (and its `groupby` in `gPost/global_post_main.py`).
- Diag scripts `test_vp.py`, `adj_precision_stud.py`, and `vp_float_case_sum()` in `solver_diag_plots.py`; the unused `calc_output_shape` import in `solver.py`.
- VP float references in CLAUDE.md and build tasks.

All computation is float64, set once at entry via `jax.config.update("jax_enable_x64", True)`. There is no precision knob.

## Consequences

- The precision study is recoverable only from git history; reviving it means restoring the wheel build and re-threading `vfloat`, not un-deleting piecemeal.
- Results schema and post-processing lose the float-precision dimension (see ADR-0001 for the rest of the schema break).
- The glossary term **VP float** is retired from CONTEXT.md.
