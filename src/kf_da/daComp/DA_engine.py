# --- Project imports ---
from kf_da.daComp.configs import *  # provides KF_Opts, DA_Opts, etc.
from kf_da.utils.utils import load_data
from kf_da.daComp.case_post_proc import post_proc_case_main
from kf_da.solver.IC_gen import Gaussian_Init, T_WARMUP
from kf_da.solver.ploting import plot_particles
from kf_da.daComp.loss_funcs import create_loss_fn, MSE_Vel
from kf_da.opti.parent_classes import LS_TR_Opt, Loss_and_Deriv_fns
from kf_da.opti.optimization import Joint_Opt
from kf_da.solver.ploting import plot_vorticity
from kf_da.velInit.AI import AI
from kf_da.solver.solver import KF_TP_Stepper, KF_Stepper, create_omega_part_gen_fn, Omega_Integrator, create_vel_trj_gen_fn

# --- Stdlib / third-party imports ---
import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import gc

# Master seed for the without-replacement permutation assigning true-IC
# snapshots to case seeds; hidden constant like T_WARMUP (ADR-0001).
MASTER_SEED = 0


def append_to_parquet(df, parquet_path):
    """
    Append a DataFrame to a Parquet file, or create it if it doesn't exist.

    Parameters
    ----------
    df : pd.DataFrame
        The new data to save.
    parquet_path : str
        Path to the Parquet file.
    """
    # If parquet doesn't exist, just write a new one
    if not os.path.exists(parquet_path):
        df.to_parquet(parquet_path, index=False)
        print(f"Created new Parquet file: {parquet_path}")
        return

    # Otherwise, load existing, concatenate, and overwrite
    existing_df = pd.read_parquet(parquet_path)
    combined = pd.concat([existing_df, df], ignore_index=True)
    combined.to_parquet(parquet_path, index=False)
    print(f"Appended data and updated {parquet_path}")

def get_tmask(T, NT, solver_dt, m_dt, loss_crit):
    idx = jnp.arange(int(T / solver_dt) + 1)

    if m_dt is None:
        idx_true = jnp.linspace(0, len(idx) - 1, NT + 1).round().astype(int)
        t_mask = jnp.zeros(len(idx), dtype=bool)
        t_mask = t_mask.at[idx_true].set(True)

        if isinstance(loss_crit, MSE_Vel):
            t_mask = t_mask.at[idx_true[0]].set(False)
    else:
        m_T = m_dt * NT

        idx_true = jnp.linspace(0, int(m_T / solver_dt), NT + 1).round().astype(int)
        if isinstance(loss_crit, MSE_Vel):
            idx_true = idx_true.at[-1].set(False)

        t_mask = jnp.zeros(len(idx), dtype=bool)
        t_mask = t_mask.at[idx_true].set(True)
        t_mask = t_mask[::-1]

    return t_mask

def warmup_snapshot_offset(t_skip):
    """Number of attractor snapshots spanning T_WARMUP; T_WARMUP must be a
    positive multiple of the snapshot spacing t_skip."""
    k = T_WARMUP / t_skip
    if T_WARMUP <= 0 or abs(k - round(k)) > 1e-9:
        raise ValueError(
            f"Warmup T_WARMUP={T_WARMUP} must be a positive multiple of t_skip={t_skip}."
        )
    return int(round(k))

def DA_exp_main(kf_opts: KF_Opts, DA_opts: DA_Opts, root) -> None:
    """
    Main entry point for running data assimilation (DA) experiments.

    The routine:
      1) Loads attractor snapshots and computes a characteristic scale.
      2) Assigns each case seed a unique true-IC snapshot via a master
         permutation (sampling without replacement, ADR-0001).
      3) For each case seed:
         - True IC = stored snapshot T_WARMUP downstream of the case's
           permuted pick; the IC guess and particle positions come from
           fold_in substreams of the case seed.
         - For each horizon T and particle count:
             * Builds the warmed particle IC (ADR-0002) and saves a PNG.
             * Generates a target trajectory via LPT advection.
             * For each NT, loss criterion, optimizer, and IC parametrization:
                 · Builds the loss function and runs the DA case.

    Parameters
    ----------
    kf_opts : KF_Opts
        Kolmogorov flow configuration.
    DA_opts : DA_Opts
        Data assimilation experiment configuration.
    """
    if DA_opts.sigma_vy > 0 and DA_opts.sigma_y == 0:
        raise ValueError("Velocity noise (sigma_vy > 0) requires position noise (sigma_y > 0).")
    if DA_opts.sigma_vy > 0 and DA_opts.part_opts.St == 0:
        raise ValueError("Velocity noise (sigma_vy > 0) requires inertial particles (St > 0).")
    particle_init = DA_opts.part_opts.particle_init
    if DA_opts.part_opts.St == 0 and isinstance(particle_init, Gaussian_Init):
        raise ValueError(
            "Gaussian particle seeding requires inertial particles (St > 0); "
            "tracer evolution ignores particle velocities."
        )
    if not isinstance(DA_opts.ic_init, AI):
        raise NotImplementedError(f"ic_init type {type(DA_opts.ic_init).__name__} is not supported")

    k_warm = warmup_snapshot_offset(kf_opts.t_skip)

    # Load attractor snapshots and compute attractor size scale
    attractor_snapshots = load_data(kf_opts)
    attractor_rad = DA_opts.ic_init.get_attractor_snaps(attractor_snapshots)

    # Master permutation over warm-start indices: case i's true IC is the
    # snapshot k_warm downstream of perm[i], so distinct cases never share a
    # true IC and case i is stable when n_cases grows (ADR-0001).
    n_eligible = attractor_snapshots.shape[0] - k_warm
    if DA_opts.n_cases > n_eligible:
        raise ValueError(
            f"n_cases={DA_opts.n_cases} exceeds the {n_eligible} attractor "
            f"snapshots eligible as warm-start states."
        )
    perm = np.random.default_rng(MASTER_SEED).permutation(n_eligible)

    os.makedirs(root, exist_ok=True)
    parquet_path = os.path.join(root, "results.parquet")

    total_cases = (
        DA_opts.n_cases
        * len(DA_opts.T_list)
        * len(DA_opts.n_particles_list)
        * len(DA_opts.NT_list)
        * len(DA_opts.crit_list)
        * len(DA_opts.optimizer_list)
        * len(DA_opts.IC_param_list)
    )
    count = 0
    # Loop over cases: one seed drives all per-case randomness (ADR-0001)
    for case_seed in range(DA_opts.n_cases):
        case_root = os.path.join(root, f"case={case_seed}")
        os.makedirs(case_root, exist_ok=True)

        idx = int(perm[case_seed])
        omega_warm_start_hat = jnp.asarray(attractor_snapshots[idx])
        omega0_hat = np.asarray(attractor_snapshots[idx + k_warm])

        U_0_path = os.path.join(case_root, "omega0_hat.npy")
        if os.path.exists(U_0_path):
            if not np.allclose(np.load(U_0_path), omega0_hat):
                raise ValueError(
                    f"Cached omega0_hat for case {case_seed} does not match "
                    f"attractor snapshot idx={idx + k_warm}; the attractor "
                    f"dataset has changed since this results tree was created."
                )
        else:
            np.save(U_0_path, omega0_hat)

        case_key = jax.random.PRNGKey(case_seed)
        guess_key = jax.random.fold_in(case_key, 0)
        part_key = jax.random.fold_in(case_key, 1)
        pos_noise_key = jax.random.fold_in(case_key, 2)
        vel_noise_key = jax.random.fold_in(case_key, 3)

        # One IC guess per case, from the case's own substream
        omega0_guess_hat, actual_norm_dist = DA_opts.ic_init(omega0_hat, guess_key)

        # Loop over time horizons
        for T in DA_opts.T_list:
            T_dir = os.path.join(case_root, f"T={T}")
            os.makedirs(T_dir, exist_ok=True)
            kf_stepper = KF_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt)
            omega_int = Omega_Integrator(kf_stepper)
            omega_trg_trj = omega_int.integrate_scan(omega0_hat, int(T/kf_opts.dt))
            np.save(os.path.join(T_dir, "omega_trg_trj.npy"), omega_trg_trj)

            # Loop over number of particles
            for npart in DA_opts.n_particles_list:
                npart_root = os.path.join(T_dir, f"np={npart}")
                os.makedirs(npart_root, exist_ok=True)

                stepper = KF_TP_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt, DA_opts.part_opts.St, DA_opts.part_opts.beta, npart)

                # Particle IC: warmed (equilibrium) or direct (gaussian), ADR-0002
                xp, yp, up, vp = particle_init.make_particle_IC(
                    npart, part_key, stepper, omega_warm_start_hat
                )
                particle_IC = (xp, yp, up, vp)

                trj_gen_fn = create_omega_part_gen_fn(jax.jit(stepper), T)
                #tuple (omega_traj, xp_traj, yp_traj, up_traj, vp_traj)
                target_trj = trj_gen_fn(omega0_hat, xp, yp, up, vp)
                xp_traj, yp_traj = target_trj[1], target_trj[2]
                up_traj, vp_traj = target_trj[3], target_trj[4]

                if DA_opts.sigma_y > 0:
                    sigma_x = DA_opts.x__y_sigma * DA_opts.sigma_y
                    x_key, y_key = jax.random.split(pos_noise_key)

                    xp_traj_DA = jnp.mod(
                        xp_traj + sigma_x * jax.random.normal(x_key, xp_traj.shape),
                        stepper.NS.L
                    )

                    yp_traj_DA = jnp.mod(
                        yp_traj + DA_opts.sigma_y * jax.random.normal(y_key, yp_traj.shape),
                        stepper.NS.L
                    )

                    fig, _ = plot_particles(xp, yp, stepper.NS.L, xp_DA=xp_traj_DA[0,:], yp_DA=yp_traj_DA[0,:], ax=None, s=20)
                    fig.savefig(os.path.join(npart_root, "particle_IC.png"))
                    plt.close(fig)
                else:
                    fig, _ = plot_particles(xp, yp, stepper.NS.L, ax=None, s=20)
                    fig.savefig(os.path.join(npart_root, "particle_IC.png"))
                    plt.close(fig)

                    xp_traj_DA = xp_traj
                    yp_traj_DA = yp_traj

                # Noisy velocity observations for inertial particles
                if DA_opts.part_opts.St > 0 and DA_opts.sigma_vy > 0:
                    sigma_vx = DA_opts.vx__vy_sigma * DA_opts.sigma_vy
                    vx_key, vy_key = jax.random.split(vel_noise_key)
                    up_traj_DA = up_traj + sigma_vx * jax.random.normal(vx_key, up_traj.shape)
                    vp_traj_DA = vp_traj + DA_opts.sigma_vy * jax.random.normal(vy_key, vp_traj.shape)
                else:
                    up_traj_DA = None
                    vp_traj_DA = None

                for NT in DA_opts.NT_list:
                    NT_root = os.path.join(npart_root, f"NT={NT}")

                    for loss_crit in DA_opts.crit_list:
                        t_mask = get_tmask(T, NT, kf_opts.dt, DA_opts.m_dt, loss_crit)
                        loss_crit.init_obj(t_mask, stepper.NS.L)
                        crit_dir = os.path.join(NT_root, f"{loss_crit}")

                        DA_part_pos_trj = (xp_traj_DA[t_mask, :], yp_traj_DA[t_mask, :])
                        if up_traj_DA is not None:
                            meas_part_vel = (up_traj_DA[t_mask, :], vp_traj_DA[t_mask, :])
                        else:
                            meas_part_vel = None

                        # For each optimizer and IC parametrization, run a DA case
                        for optimizer in DA_opts.optimizer_list:
                            if optimizer.psuedo_proj is not None:
                                kf_stepper = jax.jit(KF_Stepper(kf_opts.Re, kf_opts.n, kf_opts.NDOF, kf_opts.dt))
                                optimizer.psuedo_proj.attach_stepper(kf_stepper)

                            opt_method_dir = os.path.join(crit_dir, f"{optimizer}")

                            for IC_param in DA_opts.IC_param_list:
                                param_dir = os.path.join(opt_method_dir, f"{IC_param}")

                                # Skip if this case directory already exists
                                if os.path.isdir(param_dir):
                                    count += 1
                                    print(f"skipping case: {count}/{total_cases}")
                                    continue

                                if (DA_opts.sigma_y > 0) and (sigma_x > 0):
                                    pp_sigma = (sigma_x, DA_opts.sigma_y)
                                else:
                                    pp_sigma = None

                                checkpoint = True
                                optimize_velocity = isinstance(optimizer, Joint_Opt) and meas_part_vel is not None


                                loss_fn_and_derivs = Loss_and_Deriv_fns(loss_crit, IC_param.inv_transform, stepper, kf_stepper, target_trj, pp_sigma, DA_part_pos_trj, kf_opts.dt, T, checkpoint=checkpoint, meas_part_vel=meas_part_vel, optimize_velocity=optimize_velocity)
                                if optimizer.psuedo_proj is not None:
                                    optimizer.psuedo_proj.attach_transform(IC_param.transform, IC_param.inv_transform)

                                if isinstance(optimizer, Joint_Opt):
                                    vel_trj_gen_fn = create_vel_trj_gen_fn(kf_stepper, T)
                                    loss_fn_norm_factor = 2 * (NT * npart)
                                    if optimize_velocity:
                                        vel_sigma = (DA_opts.vx__vy_sigma * DA_opts.sigma_vy, DA_opts.sigma_vy)
                                        optimizer.set_inertial_pp_loss_fn(loss_fn_and_derivs.gen_loss_fn, loss_fn_and_derivs.PP_opt_default, pp_sigma, stepper.NS.L, vel_trj_gen_fn, t_mask, DA_part_pos_trj[0].shape, kf_opts.dt, vel_sigma, loss_fn_norm_factor)
                                    else:
                                        optimizer.set_pp_loss_fn(loss_fn_and_derivs.gen_loss_fn, loss_fn_and_derivs.PP_opt_default, pp_sigma, stepper.NS.L, vel_trj_gen_fn, t_mask, DA_part_pos_trj[0].shape, kf_opts.dt, loss_fn_norm_factor)

                                os.makedirs(param_dir)

                                results_df = pd.DataFrame({
                                                            "case_seed": [case_seed],
                                                            "T": [T],
                                                            "n_part": [npart],
                                                            "NT": [NT],
                                                            "IC_param": [f"{IC_param}"],
                                                            "init_IC_distance": [float(actual_norm_dist)],
                                                            "optimizer": [f"{optimizer}"],
                                                            "loss_crit": [f"{loss_crit}"]
                                                        })

                                _run_DA_case(target_trj, omega0_guess_hat, omega0_hat, attractor_rad, IC_param, loss_fn_and_derivs, optimizer, trj_gen_fn, particle_IC,
                                             param_dir, kf_opts.dt,
                                            t_mask, results_df,
                                            parquet_path)
                                count += 1
                                print(f"case: {count}/{total_cases}")


    return root

def _run_DA_case(
    target_trj: jnp.ndarray,
    omega0_guess_hat:  jnp.ndarray,
    omega0_hat: jnp.ndarray,
    attractor_rad: float,
    IC_param,
    loss_fn_and_derivs: Loss_and_Deriv_fns,
    optimizer: LS_TR_Opt,
    trj_gen_fn,
    particle_IC,
    save_dir,
    dt,
    t_mask,
    results_df,
    parquet_path
) -> None:
    """
    Run a single DA case for a given optimizer and loss function.

    Parameters
    ----------
    target_vel : array-like
        Target velocity trajectory (unused here but kept for symmetry/extension).
    U_0_guess : array-like
        Complex initial condition guess for the flow field.
    loss_fn : callable
        Loss function mapping real-packed IC to scalar loss.
    optimizer : Hessian_Optimizer | LBFGS
        Optimizer configuration object (used to dispatch the optimization routine).
    """
    loss_fn_and_derivs.reset_cost_count()
    Z0 = IC_param.transform(omega0_guess_hat)
    Z0_opt, opt_data = optimizer.opt_loop(Z0, loss_fn_and_derivs, IC_param.inv_transform, omega0_hat, attractor_rad)
    omega0_DA_hat = IC_param.inv_transform(Z0_opt)

    DA_trj = trj_gen_fn(omega0_DA_hat, *particle_IC)
    init_guess_trj = trj_gen_fn(omega0_guess_hat, *particle_IC)


    results_df["loss_record"] = [opt_data.loss_record]

    #saving npy files
    np.save(os.path.join(save_dir, "omega_DA_trj.npy"), np.array(DA_trj[0]))
    np.save(os.path.join(save_dir, "omega_guess_trj.npy"), np.array(init_guess_trj[0]))
    opt_data.save_data(save_dir)

    post_proc_case_main(target_trj, DA_trj, init_guess_trj, opt_data, save_dir, dt, t_mask, results_df, attractor_rad)
    append_to_parquet(results_df, parquet_path)

    #cleanup
    del  loss_fn_and_derivs
    jax.clear_caches()
    gc.collect()
