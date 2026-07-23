from __future__ import annotations

import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from kf_da.utils.plotting_utils import save_svg 
def create_results_dir():
    with open("../kf-da-configs/data_dir_plots.txt", "r") as f:
        root = f.read().rstrip("\n")
    return root

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def outlier_mask_by_loss(final_loss):
    """Boolean mask marking outliers via the modified z-score on final loss."""
    median = np.median(final_loss)
    mad = np.median(np.abs(final_loss - median))

    if mad == 0:
        # All losses identical or single sample — nothing is an outlier
        return np.zeros(final_loss.shape, dtype=bool)

    modified_z = 0.6745 * (final_loss - median) / mad
    return np.abs(modified_z) >= 3.5


def remove_outliers_by_loss(df, metric):
    loss_traces = np.vstack(df["loss_record"].to_numpy())
    final_loss = loss_traces[:, -1]

    mask = ~outlier_mask_by_loss(final_loss)
    return df.loc[mask, metric].to_numpy(), final_loss[mask]


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def m_dep_fig(cfg: dict):
    Re       = cfg["Re"]
    n        = cfg["n"]
    dt       = cfg["dt"]
    NDOF     = cfg["NDOF"]
    St       = cfg.get("St", 0)
    beta     = cfg.get("beta", 0)
    m_dt     = cfg.get("m_dt", None)
    m_targets = cfg["m_targets"]
    metric   = cfg.get("metric", "final_snap_rel_error")

    root = os.path.join(
        create_results_dir(),
        "DA-no_noise",
        f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI",
    )
    print(root)
    save_root = os.path.join(root, "global_results", "mx_v_mt")
    os.makedirs(save_root, exist_ok=True)
    df = pd.read_parquet(os.path.join(root, "results.parquet")).dropna()

    for m_target in m_targets:
        found = False
        fig = plt.figure()

        for (n_part, NT), g in df.groupby(["n_part", "NT"], sort=True):
            m = NT * 2 * n_part
            if m != m_target:
                continue
            found = True

            perf, final_loss = remove_outliers_by_loss(g, metric)
            label = f"NT={NT}, n_part={n_part}, mean={np.mean(perf):.3f}"
            plt.scatter(final_loss, perf, label=label)

        if not found:
            print(f"No runs found with m_target={m_target}")

        plt.legend()
        plt.title(f"m = {m_target} | metric = {metric}")
        plt.xlabel("loss")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.xscale("log")
        plt.ylim(0, 1)
        save_svg(mpl, fig, os.path.join(save_root, f"m={m_target}.svg"))
        plt.close(fig)


def recon_v_m_dt(cfg: dict):
    Re        = cfg["Re"]
    n         = cfg["n"]
    dt        = cfg["dt"]
    NDOF      = cfg["NDOF"]
    St        = cfg.get("St", 0)
    beta      = cfg.get("beta", 0)
    NT        = cfg["NT"]
    n_part    = cfg["n_part"]
    metric    = cfg.get("metric", "final_snap_rel_error")
    loss_crit = cfg["loss_crit"]

    pattern = re.compile(
        rf"^DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt=(\d*\.?\d+)-St={St}_beta={beta}_AI$"
    )

    base_dir = create_results_dir()
    m_dt_vals, mean_metric_vals, mean_final_loss_vals = [], [], []

    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if not os.path.isdir(full_path):
            continue
        match = pattern.match(name)
        if match is None:
            continue
        results_path = os.path.join(full_path, "results.parquet")
        if not os.path.exists(results_path):
            continue

        mdt = float(match.group(1))
        df = pd.read_parquet(results_path).dropna()
        df = df[(df["n_part"] == n_part) & (df["NT"] == NT) & (df["loss_crit"] == loss_crit)]
        if df.empty:
            continue

        metric_arr, final_loss = remove_outliers_by_loss(df, metric)
        m_dt_vals.append(mdt)
        mean_metric_vals.append(np.mean(metric_arr))
        mean_final_loss_vals.append(np.mean(final_loss))

    if not m_dt_vals:
        print("No matching runs found.")
        return

    sort_idx = np.argsort(m_dt_vals)
    m_dt_vals          = np.array(m_dt_vals)[sort_idx]
    mean_metric_vals   = np.array(mean_metric_vals)[sort_idx]
    mean_final_loss_vals = np.array(mean_final_loss_vals)[sort_idx]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    color1 = "tab:blue"
    l1 = ax1.plot(m_dt_vals, mean_metric_vals, marker="o", color=color1, label=metric)
    ax1.set_xlabel(r"$\Delta t_m$", fontsize=12)
    ax1.set_ylabel(metric, color=color1, fontsize=12)
    ax1.set_xlim(0.1, 0.8)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    l2 = ax2.plot(m_dt_vals, mean_final_loss_vals, marker="s", color=color2, label="Mean Final Loss")
    ax2.set_ylabel("Mean Final Loss", color=color2, fontsize=12)

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", frameon=False)
    plt.title(rf"Loss Criterion: {loss_crit}", fontsize=13)
    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(base_dir, "recon_v_m_dt.svg"))
    plt.close(fig)


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def recon_v_final_loss(cfg: dict):
    Re          = cfg["Re"]
    n           = cfg["n"]
    dt          = cfg["dt"]
    NDOF        = cfg["NDOF"]
    St_list     = _as_list(cfg.get("St", 0))
    beta        = cfg.get("beta", 0)
    m_dt        = cfg.get("m_dt", None)
    NT          = cfg["NT"]
    n_part      = cfg["n_part"]
    metric      = cfg.get("metric", "final_snap_rel_error")
    loss_crit   = cfg["loss_crit"]
    noise_types = _as_list(cfg.get("noise_type", "DA-no_noise"))
    ylim        = cfg.get("ylim", None)
    xlim        = cfg.get("xlim", None)

    for noise_type in noise_types:
        for St in St_list:
            root = os.path.join(
                create_results_dir(),
                noise_type,
                f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI",
            )
            results_path = os.path.join(root, "results.parquet")
            if not os.path.exists(results_path):
                print(f"recon_v_final_loss: no results at {results_path}")
                continue

            df = pd.read_parquet(results_path).dropna()
            df = df[(df["n_part"] == n_part) & (df["NT"] == NT) & (df["loss_crit"] == loss_crit)]
            if df.empty:
                print(
                    f"recon_v_final_loss: no runs for NT={NT}, n_part={n_part}, "
                    f"loss_crit={loss_crit} in {noise_type}, St={St}"
                )
                continue

            final_loss = np.vstack(df["loss_record"].to_numpy())[:, -1]
            metric_arr = df[metric].to_numpy()
            outliers   = outlier_mask_by_loss(final_loss)

            fig = plt.figure()
            plt.scatter(final_loss[~outliers], metric_arr[~outliers],
                        label=f"kept (n={np.sum(~outliers)}, mean={np.mean(metric_arr[~outliers]):.3f})")
            if outliers.any():
                plt.scatter(final_loss[outliers], metric_arr[outliers],
                            color="tab:red", marker="x", s=60,
                            label=f"outlier (n={np.sum(outliers)})")

            plt.legend()
            plt.title(f"NT={NT}, n_part={n_part} | loss_crit={loss_crit} | St={St}\n{noise_type}")
            plt.xlabel("final loss")
            plt.ylabel(metric)
            plt.xscale("log")
            if ylim is not None:
                plt.ylim(*ylim)
            if xlim is not None:
                plt.xlim(*xlim)
            plt.tight_layout()

            save_root = os.path.join(root, "global_results", "recon_v_final_loss")
            os.makedirs(save_root, exist_ok=True)
            save_svg(mpl, fig, os.path.join(save_root, f"NT={NT}_n_part={n_part}_{loss_crit}.svg"))
            plt.close(fig)


def embedding_fig(cfg: dict):
    n           = cfg["n"]
    St          = cfg.get("St", 0)
    beta        = cfg.get("beta", 0)
    m_dt        = cfg.get("m_dt", None)
    noise_type  = cfg.get("noise_type", "DA-no_noise")
    config_list = cfg["config_list"]
    dM_dict     = cfg["dM_dict"]
    metric      = cfg.get("metric", "final_snap_rel_error")
    loss_crit   = cfg.get("loss_crit", "MSE_PP")

    Re_list, dM_list, m_list, metric_list = [], [], [], []
    for Re, NDOF, dt, NT in config_list:
        root = os.path.join(
            create_results_dir(),
            noise_type,
            f"DA_Re={Re}_n={n}_dt={dt}_NDOF={NDOF}_mdt={m_dt}-St={St}_beta={beta}_AI-pinit=eq",
        ) 
        if os.path.isdir(root):
            df = pd.read_parquet(os.path.join(root, "results.parquet")).dropna()
            df = df[(df["NT"] == NT) & (df["loss_crit"] == loss_crit)]
            for n_part, df_npart in df.groupby("n_part", sort=True):
                metric_arr, _ = remove_outliers_by_loss(df_npart, metric)
                if len(metric_arr) == 0:
                    continue
                Re_list.append(Re)
                dM_list.append(dM_dict[Re])
                m_list.append(NT * n_part * 2)
                metric_list.append(np.mean(metric_arr))
    dM_arr     = np.array(dM_list)
    m_arr      = np.array(m_list)
    metric_arr = np.array(metric_list)
    Re_arr     = np.array(Re_list)

    if dM_arr.size == 0:
        print("embedding_fig: no matching data found — check config_list and NT values.")
        return

    # Build a mapping from dM -> Re for the top axis ticks
    dM_to_Re = {}
    for dM, Re in zip(dM_list, Re_list):
        dM_to_Re[dM] = Re  # last Re wins if multiple share a dM value

    dM_x = np.linspace(dM_arr.min(), dM_arr.max(), 100)

    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(dM_arr, m_arr, c=metric_arr, s=80, vmin=0.0, vmax=0.5)
    ax.plot(dM_x, dM_x,           label="immersion line")
    ax.plot(dM_x, 2 * dM_x + 1,   label="embedding line")
    fig.colorbar(sc, ax=ax).set_label("Mean Metric")
    ax.set_xlabel("IM dimension")
    ax.set_ylabel("m = NT * n_part * 2")
    ax.legend()

    # --- second x-axis at the top ---
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())  # keep axes in sync

    unique_dM = sorted(dM_to_Re.keys())
    unique_Re = [dM_to_Re[dM] for dM in unique_dM]

    ax2.set_xticks(unique_dM)
    ax2.set_xticklabels([f"{Re}" for Re in unique_Re])
    ax2.set_xlabel("Reynolds number (Re)")
    # --------------------------------

    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(create_results_dir(), noise_type, "embedding_fig.svg"))
    plt.close(fig)

    # --- reconstruction error vs measurement count, all Re on one axes ---
    unique_Re_sorted = sorted(set(Re_list))
    cmap  = plt.get_cmap("viridis", len(unique_Re_sorted))
    norm  = mpl.colors.BoundaryNorm(np.arange(len(unique_Re_sorted) + 1) - 0.5,
                                    cmap.N)

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, Re in enumerate(unique_Re_sorted):
        sel = Re_arr == Re
        m_Re      = m_arr[sel]
        metric_Re = metric_arr[sel]

        sort_idx  = np.argsort(m_Re)
        m_Re      = m_Re[sort_idx]
        metric_Re = metric_Re[sort_idx]

        dM    = dM_dict[Re]
        color = cmap(i)

        # x-axis is relative to the immersion dimension: 0 <-> m = dM
        ax.plot(m_Re - dM, metric_Re, marker="o", color=color)
        # embedding threshold 2*dM + 1 sits at dM + 1 in relative coordinates
        ax.axvline(dM + 1, color=color, ls="-", lw=1.5, alpha=0.8)

    # single immersion line, shared by all Re in relative coordinates
    ax.axvline(0.0, color="grey")

    ax.set_xlabel("m - dM")
    ax.set_ylabel(metric)
    ax.set_title(f"loss_crit = {loss_crit}")

    # line-style legend (colour is carried by the colorbar)
    style_handles = [
        mpl.lines.Line2D([], [], color="k", marker="o", label="reconstruction error"),
        mpl.lines.Line2D([], [], color="k", ls="--", lw=1, label="immersion (m = dM)"),
        mpl.lines.Line2D([], [], color="k", ls="-",  lw=1.5, label="embedding (m = 2·dM + 1)"),
    ]
    ax.legend(handles=style_handles)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        ticks=np.arange(len(unique_Re_sorted)))
    cbar.ax.set_yticklabels([f"{Re:g}" for Re in unique_Re_sorted])
    cbar.set_label("Reynolds number (Re)")

    plt.tight_layout()
    save_svg(mpl, fig, os.path.join(create_results_dir(), noise_type,
                                    "embedding_fig_recon_v_m.svg"))
    plt.close(fig)

# ---------------------------------------------------------------------------
# Config loader and registry
# ---------------------------------------------------------------------------

REGISTRY = {
    "m_dep_fig":    m_dep_fig,
    "recon_v_m_dt": recon_v_m_dt,
    "recon_v_final_loss": recon_v_final_loss,
    "embedding_fig": embedding_fig,
}


def load_cfg(path: str) -> list[tuple[str, dict]]:
    with open(path) as f:
        raw = yaml.safe_load(f)

    common = raw.get("common", {})
    run = raw.get("run", [])
    if isinstance(run, str):
        run = [run]

    tasks = []
    for fn_name in run:
        cfg = {**common, **raw.get(fn_name, {})}
        tasks.append((fn_name, cfg))
    return tasks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "kf-da-configs", "daPlotConfig.yaml")

if __name__ == "__main__":
    for fn_name, cfg in load_cfg(CONFIG_PATH):
        print(f"Running {fn_name}...")
        REGISTRY[fn_name](cfg)
