"""
Lagrangian velocity autocorrelation for tracer particles in Kolmogorov flow.

Estimates rho(tau) = E[u'(t) u'(t+tau)] / E[u'(t)^2] for the u and v particle
velocity components (u' = fluctuation about each particle's own time-mean),
then fits rho(tau) = exp(-|tau| / T_L) to recover the Lagrangian integral
timescale T_L for each component.

Ensemble: many independent draws of a flow IC from precomputed attractor
snapshots, each advecting a batch of tracer particles (St=0, beta=0) from
random start positions. For each draw, the autocorrelation is computed by
sliding the reference time t across the whole trajectory (not just t=0) and
averaged over particles and draws.
"""

import os

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from kf_da.daComp.configs import KF_Opts
from kf_da.solver.solver import KF_TP_Stepper
from kf_da.utils.create_results_dir import create_results_dir
from kf_da.utils.plotting_utils import save_svg
from kf_da.utils.utils import bilinear_sample_periodic, load_data

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Re = 100
n = 4
NDOF = 128
dt = 1e-2
T = 10.0                    # trajectory length per flow draw
St = 0.0
beta = 0.0

num_flow_draws = 30
n_part_per_draw = 64

fit_frac = 0.75              # fit exponential over tau in [0, fit_frac * T]

seed = 0


def sample_particle_ics(key, npart, L):
    k1, k2 = jax.random.split(key)
    xp = jax.random.uniform(k1, shape=(npart,), minval=0.0, maxval=L)
    yp = jax.random.uniform(k2, shape=(npart,), minval=0.0, maxval=L)
    return xp, yp


def create_lagrangian_vel_gen_fn(stepper, T):
    """
    Scans the stepper forward and, at each step, resamples the fluid velocity
    field at the particle's *current* position. This is needed because for
    tracer particles (St=0, beta=0), Tracer_Evolution never updates the
    stepper's own `up`/`vp` state -- only `xp`/`yp` are advanced using the
    fluid velocity directly, so `up`/`vp` stay frozen at their initial value.
    """
    nsteps = int(T / stepper.dt)

    def sample_vel(omega_hat, xp, yp):
        u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega_hat)
        u_grid = jnp.fft.irfft2(u_hat)
        v_grid = jnp.fft.irfft2(v_hat)
        u = bilinear_sample_periodic(u_grid, xp, yp, stepper.NS.L, stepper.NS.L)
        v = bilinear_sample_periodic(v_grid, xp, yp, stepper.NS.L, stepper.NS.L)
        return u, v

    def body(carry, _):
        omega_hat, xp, yp, up, vp = carry
        omega_hat, xp, yp, up, vp = stepper(omega_hat, xp, yp, up, vp)
        u, v = sample_vel(omega_hat, xp, yp)
        new_carry = (omega_hat, xp, yp, up, vp)
        return new_carry, (u, v)

    def gen_fn(omega0_hat, xp0, yp0):
        up0 = jnp.zeros_like(xp0)
        vp0 = jnp.zeros_like(yp0)
        carry0 = (omega0_hat, xp0, yp0, up0, vp0)
        _, (u_traj, v_traj) = jax.lax.scan(body, carry0, xs=None, length=nsteps)

        u0, v0 = sample_vel(omega0_hat, xp0, yp0)
        u_traj = jnp.concatenate([u0[None, ...], u_traj], axis=0)
        v_traj = jnp.concatenate([v0[None, ...], v_traj], axis=0)
        return u_traj, v_traj  # shape (nsteps+1, npart)

    return gen_fn


def get_particle_velocity_series(gen_fn, omega0_hat, xp0, yp0):
    up_traj, vp_traj = gen_fn(omega0_hat, xp0, yp0)
    return np.asarray(up_traj), np.asarray(vp_traj)


def windowed_autocorr(series):
    """
    series: shape (n_series, n_steps); each row is a per-particle fluctuating
    velocity time series (mean already subtracted per-row).

    Returns C(tau), shape (n_steps,): for each lag tau, averages
    fluct(t) * fluct(t+tau) over all valid reference times t and all rows.
    """
    n_steps = series.shape[1]
    C = np.empty(n_steps)
    for tau in range(n_steps):
        prod = series[:, : n_steps - tau] * series[:, tau:]
        C[tau] = prod.mean()
    return C


def exp_model(tau, T_L):
    return np.exp(-np.abs(tau) / T_L)


def fit_T_L(tau, rho, fit_frac):
    mask = tau <= fit_frac * tau[-1]
    popt, _ = curve_fit(exp_model, tau[mask], rho[mask], p0=[tau[-1] / 4])
    return float(popt[0])


def plot_component(tau, rho, T_L, label, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tau, rho, lw=1.5, label=f"{label} data")
    ax.plot(
        tau, exp_model(tau, T_L), "--", lw=2,
        label=rf"fit: $e^{{-|\tau|/T_L}}$, $T_L={T_L:.4g}$",
    )
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(rf"$\rho_{{{label}}}(\tau)$")
    ax.set_title(f"Lagrangian velocity autocorrelation ({label})")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    save_svg(mpl, fig, out_path)
    plt.close(fig)


def main():
    kf_opts = KF_Opts(
        Re=Re, n=n, NDOF=NDOF, dt=dt,
        total_T=int(1e4), min_samp_T=100, t_skip=10,
    )

    attractor_snapshots = load_data(kf_opts)
    num_snaps = attractor_snapshots.shape[0]

    stepper = KF_TP_Stepper(Re, n, NDOF, dt, St, beta, n_part_per_draw)
    gen_fn = jax.jit(create_lagrangian_vel_gen_fn(stepper, T))

    key = jax.random.PRNGKey(seed)

    u_series_all = []
    v_series_all = []

    for i in range(num_flow_draws):
        key, k_snap, k_pp = jax.random.split(key, 3)
        idx = int(jax.random.randint(k_snap, (), 0, num_snaps))
        omega0_hat = jnp.asarray(attractor_snapshots[idx])

        xp0, yp0 = sample_particle_ics(k_pp, n_part_per_draw, stepper.NS.L)

        up_traj, vp_traj = get_particle_velocity_series(
            gen_fn, omega0_hat, xp0, yp0
        )
        # transpose to (n_particles, n_steps) and remove each particle's own
        # time-mean to get the fluctuating velocity
        u_series_all.append(up_traj.T - up_traj.T.mean(axis=1, keepdims=True))
        v_series_all.append(vp_traj.T - vp_traj.T.mean(axis=1, keepdims=True))
        print(f"flow draw {i + 1}/{num_flow_draws} done")

    u_series = np.concatenate(u_series_all, axis=0)
    v_series = np.concatenate(v_series_all, axis=0)

    n_steps = u_series.shape[1]
    tau = np.arange(n_steps) * dt

    C_u = windowed_autocorr(u_series)
    C_v = windowed_autocorr(v_series)

    rho_u = C_u / C_u[0]
    rho_v = C_v / C_v[0]

    T_L_u = fit_T_L(tau, rho_u, fit_frac)
    T_L_v = fit_T_L(tau, rho_v, fit_frac)

    print(f"T_L (u) = {T_L_u:.6g}")
    print(f"T_L (v) = {T_L_v:.6g}")

    root = os.path.join(
        create_results_dir(), "autocorrelation",
        f"Re={Re}_N={NDOF}_dt={dt}_T={T}",
    )
    os.makedirs(root, exist_ok=True)

    plot_component(tau, rho_u, T_L_u, "u", os.path.join(root, "lagrangian_autocorr_u.svg"))
    plot_component(tau, rho_v, T_L_v, "v", os.path.join(root, "lagrangian_autocorr_v.svg"))


if __name__ == "__main__":
    main()
