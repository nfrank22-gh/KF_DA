"""
2D contour plots of the Stokes-number-dependent entropy term H_s(t, St)
derived in the "Stokes Number" subsection of the manuscript.

Two cases are plotted:
  - stationary initial particle velocity (eq: Hs stat)
  - Gaussian (independent) initial particle velocity (H_s = ln(Var(x(t))),
    the rho_v = rho reduction of eq: St pos cov gauss init)
"""

import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from kf_da.utils.create_results_dir import create_results_dir
from kf_da.utils.plotting_utils import save_svg

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
sigma0_sq = .1   # fluid velocity variance
T_L = 1.0         # fluid Lagrangian correlation time
sigma_v_sq = 1.0  # initial particle velocity variance (Gaussian-init case only)
sigma_v_sq = sigma0_sq

t_min, t_max, n_t = 1e-3, 3.0, 300
St_min, St_max, n_St = 1e-3, 5.0, 300

n_levels = 30
cmap = "viridis"

t_slices = [0.5, 1.0, 2.0]  # fixed times for the H_s vs St line plot


def var_x_stationary(t, St):
    """Var(x(t)) for the stationary initial particle velocity case, eq. (var x St)."""
    return 2.0 * sigma0_sq * T_L * (
        t - (
            T_L ** 3 * (1.0 - np.exp(-t / T_L))
            - St ** 3 * (1.0 - np.exp(-t / St))
        ) / (T_L ** 2 - St ** 2)
    )


def Hs_stationary(t, St):
    return np.log(var_x_stationary(t, St))


def Hs_gauss_init(t, St):
    phi = St * (1.0 - np.exp(-t / St))
    I_t = sigma0_sq * T_L * (
        T_L ** 2 * (1.0 - np.exp(-t / T_L)) - St ** 2 * (1.0 - np.exp(-t / St))
    ) / (T_L ** 2 - St ** 2)
    sigma_p_sq = sigma0_sq * T_L / (T_L + St)

    A = var_x_stationary(t, St) - 2.0 * phi * I_t + phi ** 2 * sigma_p_sq
    B = phi ** 2 * sigma_v_sq
    return np.log(A + B)


def make_St_grid():
    St_vals = np.linspace(St_min, St_max, n_St)
    # avoid the removable singularity at St = T_L
    close = np.isclose(St_vals, T_L, atol=1e-3)
    St_vals[close] *= 1.01
    return St_vals


def plot_contour(t_grid, St_grid, H, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    cs = ax.contourf(t_grid, St_grid, H, levels=n_levels, cmap=cmap)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathrm{St}$")
    ax.set_title(title)
    fig.colorbar(cs, ax=ax, label=r"$H_s(t, \mathrm{St})$")
    fig.tight_layout()
    save_svg(mpl, fig, out_path)
    plt.close(fig)


def plot_vs_St(St_vals, Hs_fn, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for t in t_slices:
        ax.plot(St_vals, Hs_fn(t, St_vals), label=f"$t={t:g}$")
    ax.set_xlabel(r"$\mathrm{St}$")
    ax.set_ylabel(r"$H_s(t, \mathrm{St})$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_svg(mpl, fig, out_path)
    plt.close(fig)


def main():
    t_vals = np.linspace(t_min, t_max, n_t)
    St_vals = make_St_grid()
    t_grid, St_grid = np.meshgrid(t_vals, St_vals)

    root = os.path.join(create_results_dir(), "St_entropy")
    os.makedirs(root, exist_ok=True)

    H_stat = Hs_stationary(t_grid, St_grid)
    plot_contour(
        t_grid, St_grid, H_stat,
        "Stokes-dependent entropy (stationary initial velocity)",
        os.path.join(root, "St_entropy_stationary.svg"),
    )

    H_gauss = Hs_gauss_init(t_grid, St_grid)
    plot_contour(
        t_grid, St_grid, H_gauss,
        "Stokes-dependent entropy (Gaussian initial velocity)",
        os.path.join(root, "St_entropy_gauss_init.svg"),
    )

    plot_vs_St(
        St_vals, Hs_stationary,
        "Stokes-dependent entropy vs St (stationary initial velocity)",
        os.path.join(root, "St_entropy_vs_St_stationary.svg"),
    )

    plot_vs_St(
        St_vals, Hs_gauss_init,
        "Stokes-dependent entropy vs St (Gaussian initial velocity)",
        os.path.join(root, "St_entropy_vs_St_gauss_init.svg"),
    )


if __name__ == "__main__":
    main()
