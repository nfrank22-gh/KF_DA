import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import config

config.update("jax_enable_x64", True)

from kf_da.solver.solver import KF_Stepper, Omega_Integrator
from kf_da.solver.ploting import plot_vorticity
from kf_da.utils.create_results_dir import create_results_dir
from kf_da.utils.plotting_utils import save_svg


def _generate_rand_IC(NDOF, key_num=0, sigma=3, kcut_frac=0.1):
    """Random vorticity IC with energy concentrated at low wavenumbers."""
    key = jax.random.PRNGKey(key_num)

    omega0 = sigma * jax.random.normal(key, (NDOF, NDOF))
    omega0 = omega0 - jnp.mean(omega0)

    omega0_hat = jnp.fft.rfft2(omega0)

    ky = jnp.fft.fftfreq(NDOF) * NDOF
    kx = jnp.fft.rfftfreq(NDOF) * NDOF
    KY, KX = jnp.meshgrid(ky, kx, indexing="ij")
    K = jnp.sqrt(KX**2 + KY**2)

    kcut = kcut_frac * (NDOF / 2)
    mask = K <= kcut

    return omega0_hat * mask


def save_vorticity_by_Re():
    """Evolve a random IC at several Re and save a vorticity snapshot figure."""
    RE_CONFIGS = [
        {"Re": 40, "NDOF": 128, "dt": 0.01},
        {"Re": 60, "NDOF": 128, "dt": 0.01},
        {"Re": 80, "NDOF": 128, "dt": 0.01},
        {"Re": 100, "NDOF": 128, "dt": 0.01},
        {"Re": 200, "NDOF": 256, "dt": 0.0025},
        {"Re": 400, "NDOF": 128, "dt": 0.0025},
    ]
    T_W = 100  # warmup time (ADR-0002 convention)
    BASE_SEED = 0  # PRNGKey(BASE_SEED + i) per Re, i = index in RE_CONFIGS
    SIGMA = 3  # random IC noise amplitude
    KCUT_FRAC = 0.1  # random IC low-pass cutoff
    N_FORCING = 4  # Kolmogorov forcing wavenumber
    TARGET_RES = 512  # plot_vorticity upsampling target

    fig, axes = plt.subplots(1, len(RE_CONFIGS), figsize=(5 * len(RE_CONFIGS), 5))

    for i, (ax, cfg) in enumerate(zip(axes, RE_CONFIGS)):
        Re, NDOF, dt = cfg["Re"], cfg["NDOF"], cfg["dt"]

        omega0_hat = _generate_rand_IC(NDOF, key_num=BASE_SEED + i, sigma=SIGMA, kcut_frac=KCUT_FRAC)

        stepper = jax.jit(KF_Stepper(Re, N_FORCING, NDOF, dt))
        integrator = Omega_Integrator(stepper)
        omega_hat = integrator.fv_integrate(omega0_hat, int(T_W / dt))
        omega = np.array(jnp.fft.irfft2(omega_hat, s=(NDOF, NDOF)))

        plot_vorticity(omega, ax=ax, target_res=TARGET_RES)
        ax.set_title(f"Re = {Re}")

    fig.tight_layout()

    save_dir = os.path.join(create_results_dir(), "general_figures")
    os.makedirs(save_dir, exist_ok=True)
    save_svg(plt, fig, os.path.join(save_dir, "vorticity_by_Re.svg"))


if __name__ == "__main__":
    save_vorticity_by_Re()
