"""
2D contour plots of the Stokes-number-dependent entropy term H_s(t, St)
derived in the "Stokes Number" subsection of the manuscript.

Three cases are plotted:
  - stationary initial particle velocity (eq: Hs stat)
  - Gaussian (independent) initial particle velocity (H_s = ln(Var(x(t))),
    the rho_v = rho reduction of eq: St pos cov gauss init)
  - fluid-matched initial particle velocity, u_p(0) = u(0)
    (H_s = ln(Var(x^f(t))), eq: Hs fluid / eq: var x fluid init)
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
sigma0_sq = 2   # fluid velocity variance
T_L = .56         # fluid Lagrangian correlation time
sigma_v_sq = 0  # initial particle velocity variance (Gaussian-init case only)

t_min, t_max, n_t = 1e-3, 3.0, 300
St_min, St_max, n_St = 1e-3, 5.0, 300

n_levels = 30
cmap = "viridis"

t_slices = [0.5, 1.0, 2.0, 3]  # fixed times for the H_s vs St line plot


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


def var_x_fluid_components(t, St):
    """Decompose Var(x^f(t)) for the fluid-matched initial-velocity case
    (u_p(0) = u(0)) into its three additive terms (eq: var x fluid init):
      comp1 = Var(x^e(t))                          -- equilibrium, fluid-driven
      comp2 = cross term between x^e(t) and d_x^f  (positive here)
      comp3 = St^2 (1 - e^{-t/St})^2 (sigma_0^2 - sigma_p^2) -- initial-velocity term
    Returns (total, comp1, comp2, comp3).
    """
    phi = St * (1.0 - np.exp(-t / St))
    J_t = sigma0_sq * St * T_L * (
        T_L * (1.0 - np.exp(-t / T_L)) - St * (1.0 - np.exp(-t / St))
    ) / (T_L ** 2 - St ** 2)
    sigma_p_sq = sigma0_sq * T_L / (T_L + St)

    comp1 = var_x_stationary(t, St)
    comp2 = 2.0 * phi * J_t
    comp3 = phi ** 2 * (sigma0_sq - sigma_p_sq)
    total = comp1 + comp2 + comp3
    return total, comp1, comp2, comp3


def Hs_fluid_init(t, St):
    total, _, _, _ = var_x_fluid_components(t, St)
    return np.log(total)


def var_x_gauss_components(t, St):
    """Decompose Var(x(t)) for the Gaussian initial-velocity case into its
    three additive terms (eq: var x gauss init):
      comp1 = Var(x^s(t))                         -- equilibrium, fluid-driven
      comp2 = cross term between x^s(t) and d_x
      comp3 = St^2 (1 - e^{-t/St})^2 (sigma_p^2 + sigma_v^2) -- initial-velocity term
    Returns (total, comp1, comp2, comp3).
    """
    phi = St * (1.0 - np.exp(-t / St))
    I_t = sigma0_sq * T_L * (
        T_L ** 2 * (1.0 - np.exp(-t / T_L)) - St ** 2 * (1.0 - np.exp(-t / St))
    ) / (T_L ** 2 - St ** 2)
    sigma_p_sq = sigma0_sq * T_L / (T_L + St)

    comp1 = var_x_stationary(t, St)
    comp2 = -2.0 * phi * I_t
    comp3 = phi ** 2 * (sigma_p_sq + sigma_v_sq)
    total = comp1 + comp2 + comp3
    return total, comp1, comp2, comp3


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


def plot_var_components_vs_St(St_vals, components_fn, suptitle, out_path):
    n = len(t_slices)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for i, t in enumerate(t_slices):
        ax = axes[i // ncols][i % ncols]
        total, comp1, comp2, comp3 = components_fn(t, St_vals)
        ax.plot(St_vals, total, label="total", color="black", linewidth=2)
        ax.plot(St_vals, comp1, label="equilibrium (fluid-driven)")
        ax.plot(St_vals, comp2, label="cross term")
        ax.plot(St_vals, comp3, label="initial-velocity term")
        ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xlabel(r"$\mathrm{St}$")
        ax.set_ylabel(r"$\mathrm{Var}(x(t))$")
        ax.set_title(f"$t={t:g}$")
        ax.legend()

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(suptitle)
    fig.tight_layout()
    save_svg(mpl, fig, out_path)
    plt.close(fig)


def main():
    t_vals = np.linspace(t_min, t_max, n_t)
    St_vals = make_St_grid()
    t_grid, St_grid = np.meshgrid(t_vals, St_vals)

    param_dir = (
        f"sigma0-{sigma0_sq:g}_TL-{T_L:g}_sigmav-{sigma_v_sq:g}"
        f"_t{t_min:g}-{t_max:g}_St{St_min:g}-{St_max:g}"
    )
    root = os.path.join(create_results_dir(), "St_entropy", param_dir)
    os.makedirs(root, exist_ok=True)

    H_stat = Hs_stationary(t_grid, St_grid)
    if False:
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

    plot_vs_St(
        St_vals, Hs_fluid_init,
        "Stokes-dependent entropy vs St (fluid-matched initial velocity)",
        os.path.join(root, "St_entropy_vs_St_fluid_init.svg"),
    )

    plot_var_components_vs_St(
        St_vals,
        var_x_gauss_components,
        "Var(x(t)) and its components vs St (Gaussian initial velocity)",
        os.path.join(root, "var_x_components_vs_St.svg"),
    )

    plot_var_components_vs_St(
        St_vals,
        var_x_fluid_components,
        r"Var(x$^f$(t)) and its components vs St (fluid-matched initial velocity)",
        os.path.join(root, "var_x_fluid_components_vs_St.svg"),
    )


if __name__ == "__main__":
    main()
