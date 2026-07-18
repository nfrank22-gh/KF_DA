import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.lines import Line2D
from DA_results_plots import remove_outliers_by_loss
from kf_da.utils.plotting_utils import save_svg


def create_results_dir():
    with open("../kf-da-configs/data_dir_plots.txt", "r") as f:
        root = f.read().rstrip("\n")
    return root

def stokes_comp():
    Re = 100
    NT = 8
    n_part_list = [20, 40, 60]

    NDOF = 128
    dt = 1e-2
    n           = 4
    beta        = 0
    m_dt        = None
    metric      = "final_snap_rel_error"
    loss_crit   = "PP_MSE"
    optimizer   = "L-BFGS_ArmBT-150"

    stokes_num_list = [0.25, 0.5, 0.75, 1]

    gauss_std = 1.0
    pinit_suffixes = {"eq": "-pinit=eq", "gauss": f"-pinit=gauss-std={gauss_std}"}
    pinit_linestyles = {"eq": "dashed", "gauss": "solid"}

    means = {(St, pinit): [] for St in stokes_num_list for pinit in pinit_suffixes}

    for St in stokes_num_list:
        for pinit, suffix in pinit_suffixes.items():
            root = os.path.join(
                create_results_dir(),
                "DA-no_noise",
                f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI{suffix}",
            )

            if os.path.isdir(root):
                df = pd.read_parquet(os.path.join(root, "results.parquet")).dropna()
                df = df[(df["NT"] == NT) & (df["loss_crit"] == loss_crit) & (df["optimizer"] == optimizer)]
                for n_part in n_part_list:
                    df_npart = df[df["n_part"] == n_part]
                    if len(df_npart) == 0:
                        continue
                    metric_arr, _ = remove_outliers_by_loss(df_npart, metric)
                    if len(metric_arr) == 0:
                        continue
                    means[(St, pinit)].append((n_part, np.mean(metric_arr)))

    pinit_handles = [
        Line2D([0], [0], color="black", linestyle=pinit_linestyles["eq"], label="Equilibrium init"),
        Line2D([0], [0], color="black", linestyle=pinit_linestyles["gauss"], label="Gaussian init"),
    ]

    fig, ax = plt.subplots()
    st_colors = dict(zip(stokes_num_list, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for St in stokes_num_list:
        for pinit in pinit_suffixes:
            pts = means[(St, pinit)]
            if not pts:
                continue
            nparts, vals = zip(*pts)
            ax.plot(nparts, vals, marker="o", color=st_colors[St], linestyle=pinit_linestyles[pinit])

    ax.set_xlabel("Number of particles")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Reconstruction error vs Stokes number (NT={NT}, Re={Re})")
    st_handles = [Line2D([0], [0], color=st_colors[St], label=f"St={St}") for St in stokes_num_list]
    st_legend = ax.legend(handles=st_handles, loc="upper right", title="Stokes number")
    ax.add_artist(st_legend)
    ax.legend(handles=pinit_handles, loc="lower right", title="Particle init")
    plt.tight_layout()
    print(os.path.join(create_results_dir(), "St_comp.svg"))
    save_svg(mpl, fig, os.path.join(create_results_dir(), "St_comp.svg"))

    means_by_npart = {(n_part, pinit): [] for n_part in n_part_list for pinit in pinit_suffixes}
    for St in stokes_num_list:
        for pinit in pinit_suffixes:
            for n_part, val in means[(St, pinit)]:
                means_by_npart[(n_part, pinit)].append((St, val))

    fig, ax = plt.subplots()
    npart_colors = dict(zip(n_part_list, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for n_part in n_part_list:
        for pinit in pinit_suffixes:
            pts = means_by_npart[(n_part, pinit)]
            if not pts:
                continue
            sts, vals = zip(*pts)
            ax.plot(sts, vals, marker="o", color=npart_colors[n_part], linestyle=pinit_linestyles[pinit])

    ax.set_xlabel("Stokes number")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Reconstruction error vs Stokes number (NT={NT}, Re={Re})")
    npart_handles = [Line2D([0], [0], color=npart_colors[n_part], label=f"n_part={n_part}") for n_part in n_part_list]
    npart_legend = ax.legend(handles=npart_handles, loc="upper right", title="Number of particles")
    ax.add_artist(npart_legend)
    ax.legend(handles=pinit_handles, loc="lower right", title="Particle init")
    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(create_results_dir(), "St_comp_vs_St.svg"))

def noise_levels():
    Re = 100
    NT = 4
    n_part_list = [20, 40, 80]

    NDOF = 128
    dt = 1e-2
    n           = 4
    St          = 0
    beta        = 0
    m_dt        = None
    metric      = "final_snap_rel_error"
    loss_crit   = "PP_MSE"
    optimizer   = "L-BFGS_ArmBT-150"

    sigma_y_list = [0, .001, .01, .1]
    x__y_sigma = 2

    means = {sigma_y: [] for sigma_y in sigma_y_list}

    for sigma_y in sigma_y_list:
        if sigma_y == 0:
            root = os.path.join(
                create_results_dir(),
                "DA-no_noise",
            )
        else:
            root = os.path.join(
                create_results_dir(),
                f"DA-sigma_y={sigma_y}--x__y_sigma={x__y_sigma}",
            )
        root = os.path.join(
            root,
            f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI",
        )

        if os.path.isdir(root):
            df = pd.read_parquet(os.path.join(root, "results.parquet")).dropna()
            df = df[(df["NT"] == NT) & (df["loss_crit"] == loss_crit) & (df["optimizer"] == optimizer)]

            for n_part in n_part_list:
                df_npart = df[df["n_part"] == n_part]
                metric_arr, _ = remove_outliers_by_loss(df_npart, metric)
                if len(metric_arr) == 0:
                    continue
                means[sigma_y].append((n_part, np.mean(metric_arr)))

    fig, ax = plt.subplots()
    for sigma_y in sigma_y_list:
        pts = means[sigma_y]
        if not pts:
            continue
        nparts, vals = zip(*pts)
        label = r"$\sigma_y=0$" if sigma_y == 0 else rf"$\sigma_y={sigma_y}$"
        ax.plot(nparts, vals, marker="o", label=label)

    ax.set_xlabel("Number of particles")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Reconstruction error vs noise (NT={NT}, Re={Re})")
    ax.legend()
    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(create_results_dir(), "noise_comp.svg"))


def optimizer_comp(
    optimizer_a: str = "L-BFGS_ArmBT-150",
    optimizer_b: str = "JO-6X-L-BFGS_ArmBT-25",
    sigma_y: float = 0.001,
    
):
    Re = 100
    NT = 4
    n_part_list = [20, 40, 80]

    NDOF = 128
    dt = 1e-2
    n         = 4
    St        = 0
    beta      = 0
    m_dt      = None
    metric    = "final_snap_rel_error"
    loss_crit = "PP_MSE"
    x__y_sigma = 2

    if sigma_y == 0:
        exp_dir = "DA-no_noise"
    else:
        exp_dir = f"DA-sigma_y={sigma_y}--x__y_sigma={x__y_sigma}"

    root = os.path.join(
        create_results_dir(),
        exp_dir,
        f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI",
    )

    if not os.path.isdir(root):
        print(f"Results directory not found: {root}")
        return

    #df = pd.read_parquet(os.path.join(root, "results.parquet")).dropna()
    df = pd.read_parquet(os.path.join(root, "results.parquet"))
    df = df.dropna(subset=[metric])
    df = df[(df["NT"] == NT) & (df["loss_crit"] == loss_crit)]

    fig, ax = plt.subplots()
    for optimizer in (optimizer_a, optimizer_b):
        df_opt = df[df["optimizer"] == optimizer]
        pts = []
        for n_part in n_part_list:
            metric_arr, _ = remove_outliers_by_loss(df_opt[df_opt["n_part"] == n_part], metric)
            metric_arr = df_opt[df_opt["n_part"] == n_part][metric].to_numpy()
            if len(metric_arr) == 0:
                continue
            pts.append((n_part, np.mean(metric_arr)))
        if not pts:
            continue
        nparts, vals = zip(*pts)
        ax.plot(nparts, vals, marker="o", label=optimizer)

    sigma_label = r"$\sigma_y=0$" if sigma_y == 0 else rf"$\sigma_y={sigma_y}$"
    ax.set_xlabel("Number of particles")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Optimizer comparison ({sigma_label}, NT={NT}, Re={Re})")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(create_results_dir(), "optimizer_comp.svg"))


if __name__ == "__main__":
    stokes_comp()
