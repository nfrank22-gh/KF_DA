"""
Run DA_exp_ctrl.py over a batch of YAML config files, spreading jobs across
the machine's GPUs.

Usage:
    uv run python main_scripts/batch_run.py config1.yaml config2.yaml ...
    uv run python main_scripts/batch_run.py --configs-file configs.txt
    uv run python main_scripts/batch_run.py --gpus 0,1,3 --jobs-per-gpu 2 *.yaml

Each config is run in its own subprocess as:
    KF_DA_CONFIG_PATH=<config> CUDA_VISIBLE_DEVICES=<gpu> uv run python main_scripts/DA_exp_ctrl.py
so it reuses DA_exp_ctrl.py's existing config-loading and results-writing logic.
GPU ids are assigned round-robin; up to --jobs-per-gpu jobs share a GPU concurrently.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def detect_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def run_one(config_path, gpu_id, log_dir, repo_root):
    log_path = log_dir / f"{Path(config_path).stem}.log"
    env = os.environ.copy()
    env["KF_DA_CONFIG_PATH"] = str(config_path)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    print(f"[start] {config_path} on GPU {gpu_id!r} -> {log_path}")
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            ["uv", "run", "python", "main_scripts/DA_exp_ctrl.py"],
            cwd=repo_root, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        )
    status = "ok" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"[done]  {config_path}: {status}")
    return config_path, result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", help="YAML config file paths")
    parser.add_argument("--configs-file", help="text file with one config path per line")
    parser.add_argument("--gpus", help="comma-separated GPU ids to use (default: autodetect all)")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, help="concurrent jobs per GPU (default: 1)")
    parser.add_argument("--log-dir", default="batch_logs", help="directory for per-config logs")
    args = parser.parse_args()

    configs = list(args.configs)
    if args.configs_file:
        with open(args.configs_file) as f:
            configs += [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not configs:
        parser.error("no config files given (pass as args or via --configs-file)")

    for c in configs:
        if not Path(c).exists():
            parser.error(f"config not found: {c}")

    if args.gpus is not None:
        gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    else:
        gpu_ids = detect_gpus()

    repo_root = Path(__file__).resolve().parent.parent
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not gpu_ids:
        print("No GPUs detected/specified; running on CPU with concurrency 1.")
        slots = [None]
    else:
        print(f"Using GPUs {gpu_ids} with {args.jobs_per_gpu} job(s) per GPU.")
        slots = [g for g in gpu_ids for _ in range(args.jobs_per_gpu)]

    max_workers = len(slots)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, config in enumerate(configs):
            gpu_id = slots[i % len(slots)]
            futures.append(pool.submit(run_one, config, gpu_id, log_dir, repo_root))
        for fut in futures:
            results.append(fut.result())

    failed = [c for c, rc in results if rc != 0]
    print(f"\n{len(results) - len(failed)}/{len(results)} configs succeeded.")
    if failed:
        print("Failed configs:")
        for c in failed:
            print(f"  {c}")
        sys.exit(1)


if __name__ == "__main__":
    main()
