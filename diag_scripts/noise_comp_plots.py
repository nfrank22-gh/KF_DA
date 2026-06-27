import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from DA_results_plots import remove_outliers_by_loss
from kf_da.utils.plotting_utils import save_svg


def create_results_dir():
    with open("../kf-da-configs/data_dir_plots.txt", "r") as f:
        root = f.read().rstrip("\n")
    return root

def stokes_comp():
    Re = 100
    NT = 4
    n_part_list = [20, 40, 80]

    NDOF = 128
    dt = 1e-2
    n           = 4
    beta        = 0
    m_dt        = None
    metric      = "final_snap_rel_error"
    loss_crit   = "PP_MSE"
    optimizer   = "L-BFGS_ArmBT-150"

    stokes_num_list = [0, 0.01, .25, 0.5, 1, 2, 5]

    means = {St: [] for St in stokes_num_list}

    for St in stokes_num_list:
        root = os.path.join(
            create_results_dir(),
            "DA-no_noise",
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
                means[St].append((n_part, np.mean(metric_arr)))

    fig, ax = plt.subplots()
    for St in stokes_num_list:
        pts = means[St]
        if not pts:
            continue
        nparts, vals = zip(*pts)
        ax.plot(nparts, vals, marker="o", label=f"St={St}")

    ax.set_xlabel("Number of particles")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"Reconstruction error vs Stokes number (NT={NT}, Re={Re})")
    ax.legend()
    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(create_results_dir(), "St_comp.svg"))

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
    optimizer_comp()
