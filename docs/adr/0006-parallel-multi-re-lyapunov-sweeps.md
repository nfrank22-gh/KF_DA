# 0006: Parallel multi-Re Lyapunov sweeps via subprocess-per-GPU

## Status

Accepted

## Context

`ly_exp.py` computes the Lyapunov spectrum for a single Re, read from
`../kf-da-configs/lyExpConfig.yaml`. Running a sweep over several Re values
meant hand-invoking the script repeatedly, serially, on one GPU.

JAX binds to whichever GPUs are visible at process start (via
`CUDA_VISIBLE_DEVICES`), so true concurrent execution across GPUs requires
separate OS processes, not in-process device placement. `batch_run.py`
already established this pattern for `DA_exp_ctrl.py`: autodetect GPUs via
`nvidia-smi`, round-robin one job per GPU, run each as a subprocess with
`CUDA_VISIBLE_DEVICES` set, log to a per-job file, and report a pass/fail
summary at the end.

## Decision

Extend `ly_exp.py` itself (no new script) to support a multi-Re sweep:

- `lyExpConfig.yaml`'s `Re` key accepts either a scalar int (existing
  behavior, unchanged) or a list of ints (sweep).
- When `Re` is a list, `ly_exp.py`'s entry point does not run the Lyapunov
  computation itself. Instead it autodetects GPUs (reusing `batch_run.py`'s
  `detect_gpus` logic), assigns each Re round-robin to a GPU — one job per
  GPU, no `jobs-per-gpu`/`--gpus` config knobs — and launches one subprocess
  per Re: `uv run python main_scripts/ly_exp.py` with env
  `CUDA_VISIBLE_DEVICES=<gpu>` and `LY_EXP_RE_OVERRIDE=<re>`.
- Each subprocess loads the same YAML (still containing the full `Re` list),
  but `LY_EXP_RE_OVERRIDE` short-circuits: it overrides `config["Re"]` to
  that single value and skips orchestration, running the normal single-Re
  path. Existing output paths already key on Re
  (`Re=…_NDOF=…_dt=…_T=…/`), so concurrent runs can't collide.
- No GPUs detected → fall back to running the Re values as subprocesses
  sequentially on CPU (concurrency 1), matching `batch_run.py`'s fallback.
- Subprocess stdout/stderr is redirected to a per-Re log file (e.g.
  `ly_batch_logs/Re=<value>.log`); the parent prints `[start]`/`[done]`
  lines and a final summary.
- A failing Re run does not stop the others; the parent waits for all jobs
  and exits non-zero with a list of failed Re values if any failed.
- No cross-Re aggregation output (e.g. LLE-vs-Re plot). Out of scope for
  this change — each Re's output stays exactly as it is today.

## Consequences

- GPU-detection/round-robin/log-redirect logic is now duplicated between
  `batch_run.py` and `ly_exp.py` (a few dozen lines). Acceptable for now;
  worth factoring into a shared helper if a third caller appears.
- The `Re` config key is now polymorphic (int or list of int), which is a
  minor deviation from the otherwise-flat `lyExpConfig.yaml` schema; the
  single-Re path is unchanged so existing configs keep working unmodified.
- Multi-Re sweeps depend on `nvidia-smi`/subprocess re-invocation, same
  operational assumptions as `batch_run.py` (requires `uv` on PATH, GPUs
  visible to the launching shell).
