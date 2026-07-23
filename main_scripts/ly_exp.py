import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import yaml

from kf_da.utils.utils import load_data
from kf_da.utils.create_results_dir import create_results_dir
from kf_da.daComp import KF_Opts
from kf_da.solver.solver import KF_Stepper


def kaplan_yorke_dimension(lyap: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Kaplan-Yorke (Lyapunov) dimension from a 1D array of Lyapunov exponents.

    - Accepts exponents in any order (will sort descending).
    - Returns a scalar jnp.ndarray (float).
    """
    lyap = jnp.asarray(lyap).reshape(-1)
    lam = jnp.sort(lyap)[::-1]

    csum = jnp.cumsum(lam)
    pos = csum > 0.0
    k = jnp.sum(pos).astype(jnp.int32)  # k in {0,...,n}

    n = lam.shape[0]

    def case_k0(_):
        return jnp.array(0.0, dtype=lam.dtype)

    def case_kn(_):
        return jnp.array(n, dtype=lam.dtype)

    def case_mid(_):
        Sk = csum[k - 1]
        lam_next = lam[k]
        return k + Sk / jnp.abs(lam_next)

    return jax.lax.cond(
        k == 0,
        case_k0,
        lambda _: jax.lax.cond(k == n, case_kn, case_mid, operand=None),
        operand=None,
    )


def push_orthonormal_matrix_variation(stepper, u_0, Y_0, n, k: int):
    """
    Propagate orthonormal columns Y under the linearized dynamics of `stepper`,
    performing a QR re-orthonormalization every `k` steps.

    Args
    ----
    stepper : callable
        Nonlinear step: u_next = stepper(u). Must be JAX-traceable.
    u_0 : array
        Initial state.
    Y_0 : array, shape (dim, r)
        Initial basis (columns). Will be orthonormalized at start.
    n : int
        Number of steps to run.
    k : int
        Period for QR re-orthonormalization (k >= 1).

    Returns
    -------
    growth_trj : array, shape (n, r)
        Per-step growth factors. On QR steps, equals diag(R).
        On non-QR steps, filled with ones.
    """
    Q0, _ = jnp.linalg.qr(Y_0)

    @jax.jit
    def scan_fn(carry, t):
        u, Y = carry

        u_next, jvp_fn = jax.linearize(stepper, u)
        Y_prop = jax.vmap(jvp_fn, in_axes=-1, out_axes=-1)(Y)

        do_qr = (t + 1) % k == 0  # QR on steps k, 2k, 3k, ...

        def qr_branch(Yp):
            Q, R = jnp.linalg.qr(Yp, mode="reduced")
            growth = jnp.diag(R)
            return Q, growth

        def passthrough_branch(Yp):
            r = Yp.shape[-1]
            return Yp, jnp.ones((r,), dtype=Yp.dtype)

        Y_next, growth = jax.lax.cond(do_qr, qr_branch, passthrough_branch, Y_prop)

        return (u_next, Y_next), growth

    steps = jnp.arange(n, dtype=jnp.int32)

    (_, _Y_final), growth_trj = jax.lax.scan(
        scan_fn,
        (u_0, Q0),
        steps,
        length=n,
    )
    return growth_trj


def plot_lyapunov_spectrum(lyap_sorted, out_path):
    idx = np.arange(1, len(lyap_sorted) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    ax.plot(idx, lyap_sorted, marker="o", linestyle="-")
    ax.set_xlabel("Index")
    ax.set_ylabel(r"Lyapunov exponent $\lambda_i$")
    ax.set_title("Lyapunov spectrum")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def detect_gpus():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def run_one_re(re_val, gpu_id, log_dir, repo_root):
    log_path = log_dir / f"Re={re_val}.log"
    env = os.environ.copy()
    env["LY_EXP_RE_OVERRIDE"] = str(re_val)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

    print(f"[start] Re={re_val} on GPU {gpu_id!r} -> {log_path}")
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            ["uv", "run", "python", "main_scripts/ly_exp.py"],
            cwd=repo_root, env=env, stdout=log_f, stderr=subprocess.STDOUT,
        )
    status = "ok" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"[done]  Re={re_val}: {status}")
    return re_val, result.returncode


def run_re_sweep(re_values):
    gpu_ids = detect_gpus()
    repo_root = Path(__file__).resolve().parent.parent
    log_dir = Path("ly_batch_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if not gpu_ids:
        print("No GPUs detected; running Re sweep sequentially on CPU.")
        slots = [None]
    else:
        print(f"Using GPUs {gpu_ids}, one Re per GPU.")
        slots = gpu_ids

    max_workers = len(slots)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(run_one_re, re_val, slots[i % len(slots)], log_dir, repo_root)
            for i, re_val in enumerate(re_values)
        ]
        for fut in futures:
            results.append(fut.result())

    failed = [re_val for re_val, rc in results if rc != 0]
    print(f"\n{len(results) - len(failed)}/{len(results)} Re values succeeded.")
    if failed:
        print("Failed Re values:")
        for re_val in failed:
            print(f"  {re_val}")
        sys.exit(1)


def ly_exp_main():
    with open("../kf-da-configs/lyExpConfig.yaml") as f:
        config = yaml.safe_load(f)["config"]

    re_override = os.environ.get("LY_EXP_RE_OVERRIDE")
    if re_override is not None:
        config["Re"] = int(re_override) if float(re_override).is_integer() else float(re_override)
    elif isinstance(config["Re"], list):
        run_re_sweep(config["Re"])
        return

    kf_opts = KF_Opts(
        Re=config["Re"],
        n=config["n"],
        NDOF=config["NDOF"],
        dt=float(config["dt"]),
        total_T=config["total_T"],
        min_samp_T=config["min_samp_T"],
        t_skip=float(config["t_skip"]),
    )

    seed = config["seed"]
    r = config["n_exponents"]
    T = config["T"]
    T_skip = config["T_skip"]

    root = os.path.join(
        create_results_dir(),
        "Ly_Exps",
        f"Re={kf_opts.Re}_NDOF={kf_opts.NDOF}_dt={kf_opts.dt}_T={T}",
    )
    os.makedirs(root, exist_ok=True)

    attractor_snapshots = load_data(kf_opts)
    key = jax.random.PRNGKey(seed)

    num_snapshots = attractor_snapshots.shape[0]
    idx = jax.random.randint(key, shape=(), minval=0, maxval=num_snapshots)

    U_0 = attractor_snapshots[idx, :]
    state_shape = U_0.shape
    U_0 = U_0.reshape(-1)

    stepper_raw = KF_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt)
    stepper = lambda x: stepper_raw(x.reshape(*state_shape)).reshape(-1)
    n = U_0.shape[0]

    A = jax.random.normal(key, (n, r), dtype=U_0.dtype)
    Y_0, _ = jnp.linalg.qr(A)

    n_steps = int(T / kf_opts.dt)
    n_skip = int(T_skip / kf_opts.dt)

    growth_trj = push_orthonormal_matrix_variation(stepper, U_0, Y_0, n_steps, n_skip)

    lyapunov_spectrum = jnp.sum(jnp.log(jnp.abs(growth_trj)), axis=0) / T

    lyap_np = np.asarray(lyapunov_spectrum)
    lyap_sorted = np.sort(lyap_np)[::-1]

    LLE = float(lyap_sorted[0])
    KY_dim = float(kaplan_yorke_dimension(lyap_sorted))

    out_file = os.path.join(root, "lyapunov_spectrum.txt")
    with open(out_file, "w") as f:
        f.write("# Lyapunov analysis\n")
        f.write(f"# Re        = {kf_opts.Re}\n")
        f.write(f"# NDOF      = {kf_opts.NDOF}\n")
        f.write(f"# dt        = {kf_opts.dt}\n")
        f.write(f"# T         = {T}\n")
        f.write(f"# r         = {r}\n")
        f.write("\n")

        f.write(f"LLE = {LLE:.8e}\n")
        f.write(f"KY_dim = {KY_dim:.8f}\n")
        f.write("\n")

        f.write("# Lyapunov exponents (sorted, descending)\n")
        for i, le in enumerate(lyap_sorted):
            f.write(f"{i:3d}  {le:.8e}\n")

    plot_file = os.path.join(root, "lyapunov_spectrum.png")
    plot_lyapunov_spectrum(lyap_sorted, plot_file)

    print(f"Saved Lyapunov results to: {out_file}")
    print(f"Saved Lyapunov spectrum plot to: {plot_file}")
    print(f"LLE = {LLE}")
    print(f"KY_dim = {KY_dim}")


if __name__ == "__main__":
    ly_exp_main()
