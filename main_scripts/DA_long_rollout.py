"""Cycled (segmented) data assimilation over a long time horizon (ADR-0005).

Runs a chain of DA optimizations over ``n_segments`` consecutive assimilation
windows, each of length ``T_dict[Re]`` (the Lyapunov timescale, same as the main
experiments). Each segment's analysis is rolled forward through its window and
the resulting forecast state seeds the next segment's IC guess (cycled 4D-Var).

Tracer particles only (St = 0), no observation noise. Config is read from a YAML
file; ``mode`` selects run / post / run_post.

    python main_scripts/DA_long_rollout.py
"""

import os
import shutil

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import yaml
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation

from kf_da.daComp import KF_Opts, MSE_PP
from kf_da.daComp.DA_engine import get_tmask, warmup_snapshot_offset
from kf_da.opti.parent_classes import Loss_and_Deriv_fns
from kf_da.opti import ArmijoLineSearch, BFGS
from kf_da.icParam import Fourier_Param
from kf_da.velInit import AI
from kf_da.solver.IC_gen import Equilibrium_Init
from kf_da.solver.solver import (
    KF_Stepper,
    KF_TP_Stepper,
    Omega_Integrator,
    create_omega_part_gen_fn,
)
from kf_da.utils.utils import load_data
from kf_da.utils.create_results_dir import create_results_dir

# Window length per Re (Lyapunov timescale), identical to DA_exp_ctrl.py.
T_DICT = {200: 3.2, 100: 3.2, 80: 3.6, 60: 4.2, 40: 0.43}

# Hardcoded optimizer / parametrization (matches DA_exp_ctrl's style, ADR-0005).
N_FORCING = 4          # forcing wavenumber n
BFGS_ITS = 150
ST = 0.0               # tracers only
BETA = 0.0
MIN_SAMP_T = 100       # attractor dataset sampling interval (for load_data)
T_SKIP = 1


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config():
    yaml_root = os.environ.get(
        "KF_DA_LONG_ROLLOUT_CONFIG_PATH",
        "../kf-da-configs/daLongRolloutConfig.yaml",
    )
    with open(yaml_root) as f:
        cfg = yaml.safe_load(f)
    cfg["_yaml_path"] = yaml_root
    return cfg


def run_name(cfg):
    return (
        f"Re={cfg['Re']}_NDOF={cfg['NDOF']}_nseg={cfg['n_segments']}"
        f"_np={cfg['n_particles']}_NT={cfg['NT']}_seed={cfg['seed']}"
    )


def output_dir(cfg):
    return os.path.join(create_results_dir(), "LongRollout", run_name(cfg))


# --------------------------------------------------------------------------- #
# Run: cycled DA
# --------------------------------------------------------------------------- #
def run_experiment(cfg, out_dir):
    Re, NDOF, dt = cfg["Re"], cfg["NDOF"], float(cfg["dt"])
    n_seg = int(cfg["n_segments"])
    npart = int(cfg["n_particles"])
    NT = int(cfg["NT"])
    seed = int(cfg["seed"])

    T_seg = T_DICT[Re]
    nsteps_seg = int(T_seg / dt)

    os.makedirs(out_dir, exist_ok=True)

    # --- Attractor snapshots, true IC, warm start ---
    total_T = int(float(cfg.get("total_T", 1e4)))
    kf_opts = KF_Opts(Re=Re, n=N_FORCING, NDOF=NDOF, dt=dt,
                      total_T=total_T, min_samp_T=MIN_SAMP_T, t_skip=T_SKIP)
    attractor = load_data(kf_opts)

    ic_init = AI(min_norm=float(cfg["min_norm"]), max_norm=float(cfg["max_norm"]))
    attractor_rad = float(ic_init.get_attractor_snaps(attractor))

    k_warm = warmup_snapshot_offset(T_SKIP)
    n_eligible = attractor.shape[0] - k_warm
    # seed picks the true-IC snapshot; true IC lies k_warm downstream of the
    # warm-start snapshot so Equilibrium_Init can co-evolve particles (ADR-0002).
    idx = int(np.random.default_rng(seed).integers(n_eligible))
    omega_warm_start_hat = jnp.asarray(attractor[idx])
    true_IC_hat = jnp.asarray(attractor[idx + k_warm])

    case_key = jax.random.PRNGKey(seed)
    guess_key = jax.random.fold_in(case_key, 0)
    part_key = jax.random.fold_in(case_key, 1)

    # --- Steppers ---
    stepper_tp = KF_TP_Stepper(Re, N_FORCING, NDOF, dt, ST, BETA, npart)
    stepper_tp_j = jax.jit(stepper_tp)
    kf_stepper = jax.jit(KF_Stepper(Re, N_FORCING, NDOF, dt))
    omega_int = Omega_Integrator(kf_stepper)
    ref_seg_gen = create_omega_part_gen_fn(stepper_tp_j, T_seg)

    # --- Particle IC (once, t=0), continuous over the whole horizon (ADR-0005) ---
    xp0, yp0, up0, vp0 = Equilibrium_Init().make_particle_IC(
        npart, part_key, stepper_tp, omega_warm_start_hat
    )

    IC_param = Fourier_Param(NDOF, NDOF // 2, beta=0.0, Re=Re)
    loss_crit = MSE_PP()
    t_mask = get_tmask(T_seg, NT, dt, None, loss_crit)
    loss_crit.init_obj(t_mask, stepper_tp.NS.L)

    ls = ArmijoLineSearch(alpha_init=1.0, rho=0.25, c=1e-4, max_iters=5)

    # Accumulators (host-side numpy to bound device memory)
    ref_omega_chunks, DA_omega_chunks = [], []
    ref_xp_chunks, ref_yp_chunks = [], []
    omega0_DA_per_seg, omega0_ref_per_seg = [], []
    loss_records, grad_records = {}, {}

    # Reference carry (flow + particles), advanced segment by segment
    ref_carry = (true_IC_hat, xp0, yp0, up0, vp0)
    guess_hat = None  # set for segment 0 below

    for k in range(n_seg):
        omega0_ref_k = ref_carry[0]

        # --- Reference rollout for this window (flow + tracers) ---
        target_seg = ref_seg_gen(*ref_carry)  # each length nsteps_seg+1
        omega_ref_seg, xp_ref_seg, yp_ref_seg, up_ref_seg, vp_ref_seg = target_seg

        # --- IC guess: attractor draw for seg 0, else forecast handoff ---
        if k == 0:
            guess_hat, _ = ic_init(omega0_ref_k, guess_key)
            guess_seg0_hat = guess_hat

        # --- Build loss over this window (masked reference positions = obs) ---
        meas_part_pos = (xp_ref_seg[t_mask, :], yp_ref_seg[t_mask, :])
        loss_fns = Loss_and_Deriv_fns(
            loss_crit, IC_param.inv_transform, stepper_tp, kf_stepper,
            target_seg, None, meas_part_pos, dt, T_seg, checkpoint=True,
        )
        loss_fns.reset_cost_count()

        optimizer = BFGS(ls=ls, its=BFGS_ITS, max_mem=20, eps_H=1e-10,
                         print_loss=True)

        # --- Optimize the IC at the window start ---
        Z0 = IC_param.transform(guess_hat)
        Z0_opt, opt_data = optimizer.opt_loop(
            Z0, loss_fns, IC_param.inv_transform, omega0_ref_k, attractor_rad
        )
        omega0_DA_k = IC_param.inv_transform(Z0_opt)

        # --- Analysis rollout through the window; end state = next guess ---
        omega_DA_seg = omega_int.integrate_scan(omega0_DA_k, nsteps_seg)
        guess_hat = omega_DA_seg[-1]

        # --- Accumulate (drop duplicated boundary frame except on last seg) ---
        keep = slice(None) if k == n_seg - 1 else slice(0, nsteps_seg)
        ref_omega_chunks.append(np.asarray(omega_ref_seg[keep]))
        DA_omega_chunks.append(np.asarray(omega_DA_seg[keep]))
        ref_xp_chunks.append(np.asarray(xp_ref_seg[keep]))
        ref_yp_chunks.append(np.asarray(yp_ref_seg[keep]))
        omega0_DA_per_seg.append(np.asarray(omega0_DA_k))
        omega0_ref_per_seg.append(np.asarray(omega0_ref_k))
        loss_records[f"seg_{k}"] = np.asarray(opt_data.loss_record)
        grad_records[f"seg_{k}"] = np.asarray(opt_data.grad_norm_record)

        # --- Advance the reference to the next window start ---
        ref_carry = (omega_ref_seg[-1], xp_ref_seg[-1], yp_ref_seg[-1],
                     up_ref_seg[-1], vp_ref_seg[-1])

        print(f"[segment {k + 1}/{n_seg}] final loss "
              f"{opt_data.loss_record[opt_data.loss_record != 0][-1]:.4e}")

    # --- Concatenate full-horizon trajectories ---
    omega_ref_trj = np.concatenate(ref_omega_chunks, axis=0)
    omega_DA_trj = np.concatenate(DA_omega_chunks, axis=0)
    xp_ref_trj = np.concatenate(ref_xp_chunks, axis=0)
    yp_ref_trj = np.concatenate(ref_yp_chunks, axis=0)

    # rel spectral L2 error vs time (matches case_post_proc.rel_error)
    error_vs_time = np.linalg.norm(
        omega_DA_trj - omega_ref_trj, axis=(1, 2)
    ) / attractor_rad

    # --- Save everything (ADR-0005) ---
    np.save(os.path.join(out_dir, "omega_ref_trj.npy"), omega_ref_trj)
    np.save(os.path.join(out_dir, "omega_DA_trj.npy"), omega_DA_trj)
    np.save(os.path.join(out_dir, "xp_ref_trj.npy"), xp_ref_trj)
    np.save(os.path.join(out_dir, "yp_ref_trj.npy"), yp_ref_trj)
    np.save(os.path.join(out_dir, "omega0_DA_per_segment.npy"),
            np.stack(omega0_DA_per_seg))
    np.save(os.path.join(out_dir, "omega0_ref_per_segment.npy"),
            np.stack(omega0_ref_per_seg))
    np.save(os.path.join(out_dir, "omega0_guess_seg0.npy"),
            np.asarray(guess_seg0_hat))
    np.save(os.path.join(out_dir, "error_vs_time.npy"), error_vs_time)
    np.savez(os.path.join(out_dir, "loss_records.npz"), **loss_records)
    np.savez(os.path.join(out_dir, "grad_norm_records.npz"), **grad_records)
    np.savez(
        os.path.join(out_dir, "meta.npz"),
        dt=dt, T_seg=T_seg, n_segments=n_seg, nsteps_seg=nsteps_seg,
        attractor_rad=attractor_rad, NDOF=NDOF, Re=Re,
        n_particles=npart, NT=NT, seed=seed,
    )
    shutil.copyfile(cfg["_yaml_path"], os.path.join(out_dir, "config.yaml"))
    print(f"Saved run data to {out_dir}")


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def _load_meta(out_dir):
    m = np.load(os.path.join(out_dir, "meta.npz"))
    return {k: m[k].item() for k in m.files}


def plot_error_vs_time(out_dir):
    meta = _load_meta(out_dir)
    err = np.load(os.path.join(out_dir, "error_vs_time.npy"))
    dt, nsteps_seg, n_seg = meta["dt"], meta["nsteps_seg"], meta["n_segments"]
    t = np.arange(err.shape[0]) * dt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, err, lw=1.2, color="C0")
    for k in range(1, n_seg):  # segment boundaries
        ax.axvline(k * nsteps_seg * dt, color="0.7", lw=0.8, ls="--")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\|\hat\omega_{DA}-\hat\omega_{ref}\|/r_{attr}$")
    ax.set_title("Cycled DA reconstruction error vs time")
    ax.set_yscale("log")
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_vs_time.png"), dpi=150)
    plt.close(fig)


def plot_loss_convergence(out_dir):
    meta = _load_meta(out_dir)
    n_seg = meta["n_segments"]
    losses = np.load(os.path.join(out_dir, "loss_records.npz"))

    fig, ax = plt.subplots(figsize=(9, 4))
    offset = 0
    cmap = plt.get_cmap("viridis")
    for k in range(n_seg):
        rec = losses[f"seg_{k}"]
        rec = rec[rec != 0]  # BFGS early-stop truncation leaves trailing zeros
        it = np.arange(rec.shape[0]) + offset
        ax.plot(it, rec, color=cmap(k / max(n_seg - 1, 1)), lw=1.0)
        if k > 0:
            ax.axvline(offset, color="0.85", lw=0.6)
        offset += rec.shape[0]
    ax.set_xlabel("cumulative optimizer iteration (across segments)")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title("Per-segment optimizer convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_convergence.png"), dpi=150)
    plt.close(fig)


def make_video(out_dir, max_frames=600, trail_len=15):
    meta = _load_meta(out_dir)
    L = 2 * np.pi
    omega_ref = np.load(os.path.join(out_dir, "omega_ref_trj.npy"))
    omega_DA = np.load(os.path.join(out_dir, "omega_DA_trj.npy"))
    xp_ref = np.load(os.path.join(out_dir, "xp_ref_trj.npy"))
    yp_ref = np.load(os.path.join(out_dir, "yp_ref_trj.npy"))

    # spectral -> physical vorticity
    w_ref = np.fft.irfft2(omega_ref, axes=(-2, -1))
    w_DA = np.fft.irfft2(omega_DA, axes=(-2, -1))
    w_err = w_DA - w_ref

    n_frames = w_ref.shape[0]
    skip = max(1, n_frames // max_frames)
    frames = list(range(0, n_frames, skip))

    npart = xp_ref.shape[1]

    norm = colors.TwoSlopeNorm(vmin=-10, vcenter=0.0, vmax=10)
    err_norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0.0, vmax=2)
    fig, (axr, axd, axe, axp) = plt.subplots(1, 4, figsize=(19, 5))
    im_r = axr.imshow(w_ref[0], origin="lower", extent=[0, L, 0, L],
                      cmap="RdBu_r", norm=norm, aspect="equal")
    im_d = axd.imshow(w_DA[0], origin="lower", extent=[0, L, 0, L],
                      cmap="RdBu_r", norm=norm, aspect="equal")
    im_e = axe.imshow(w_err[0], origin="lower", extent=[0, L, 0, L],
                      cmap="RdBu_r", norm=err_norm, aspect="equal")
    axr.set_title("reference")
    axd.set_title("DA")
    axe.set_title("error (DA - reference)")
    axp.set_title("true particle positions")
    for a in (axr, axd, axe):
        a.set_xticks([]); a.set_yticks([])
    axp.set_xlim(0, L); axp.set_ylim(0, L)
    axp.set_aspect("equal")
    axp.set_xticks([]); axp.set_yticks([])
    axp.set_facecolor("white")
    fig.colorbar(im_d, ax=(axr, axd), shrink=0.7)
    fig.colorbar(im_e, ax=axe, shrink=0.7)
    sup = fig.suptitle("t = 0.00")

    # comet-trail scatter for particles: newest points opaque, older points fade out.
    # Fixed-size point cloud (trail_len blocks of npart), only offsets/alpha move per frame.
    scat = axp.scatter(np.zeros(trail_len * npart), np.zeros(trail_len * npart),
                        s=10, linewidths=0)
    age_alpha = np.linspace(1.0, 0.05, trail_len)
    base_rgba = np.array(colors.to_rgba("black"))
    face_colors = np.tile(base_rgba, (trail_len * npart, 1))
    for i in range(trail_len):
        face_colors[i * npart:(i + 1) * npart, 3] = age_alpha[i]
    scat.set_facecolor(face_colors)

    dt = meta["dt"]

    def update(fi):
        im_r.set_data(w_ref[fi])
        im_d.set_data(w_DA[fi])
        im_e.set_data(w_err[fi])

        trail_idx = [max(fi - j * skip, 0) for j in range(trail_len)]
        xs = xp_ref[trail_idx].ravel()
        ys = yp_ref[trail_idx].ravel()
        scat.set_offsets(np.column_stack([xs, ys]))

        sup.set_text(f"t = {fi * dt:.2f}")
        return im_r, im_d, im_e, scat, sup

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(os.path.join(out_dir, "vorticity_side_by_side.mp4"),
              writer="ffmpeg", fps=30, dpi=130)
    plt.close(fig)


def post_process(out_dir):
    plot_error_vs_time(out_dir)
    plot_loss_convergence(out_dir)
    make_video(out_dir)
    print(f"Wrote figures/video to {out_dir}")


# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()
    out_dir = output_dir(cfg)
    mode = cfg.get("mode", "run_post")

    if mode in ("run", "run_post"):
        run_experiment(cfg, out_dir)
    if mode in ("post", "run_post"):
        post_process(out_dir)
    if mode not in ("run", "post", "run_post"):
        raise ValueError(f"Unknown mode: {mode!r} (expected run | post | run_post)")


if __name__ == "__main__":
    main()
