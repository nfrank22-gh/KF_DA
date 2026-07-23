"""
Re-generate the standard DA case plots (u_N comparisons, particle tracks,
error-vs-time, optimization convergence) from .npy data pulled out of a
case's results folder.

Usage:
    python diag_scripts/replot_case.py /path/to/assembled_case_folder

The folder must contain (copied verbatim from the case's results dir, see
case_post_proc.post_proc_case_main and DA_engine._run_DA_case):
    omega_trg_trj.npy
    omega_DA_trj.npy
    omega_guess_trj.npy
    xp_trg.npy, yp_trg.npy, xp_DA.npy, yp_DA.npy
    xp_DA_obs.npy, yp_DA_obs.npy, xp_DA_drift.npy, yp_DA_drift.npy,
    xp_DA_reset.npy, yp_DA_reset.npy (ADR-0007, observer track)
    t_mask.npy
    vel_error.npy, time_axis.npy
    loss_record.npy, grad_norm_record.npy, alpha_gTp_record.npy, IC_error_record.npy

Plots are written into that same folder.
"""

import argparse
import os
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from kf_da.daComp.case_post_proc import (
    plot_convergence,
    plot_field_error_row,
    plot_particle_tracks,
    plot_vel_error_vs_time,
    plot_vort_comp,
)


def _load(case_dir, name):
    return np.load(os.path.join(case_dir, name))


def replot_case(case_dir):
    omega_trg_hat = _load(case_dir, "omega_trg_trj.npy")
    omega_DA_hat = _load(case_dir, "omega_DA_trj.npy")
    omega_guess_hat = _load(case_dir, "omega_guess_trj.npy")

    omega_trg = jnp.fft.irfft2(omega_trg_hat, axes=(-2, -1))
    omega_DA = jnp.fft.irfft2(omega_DA_hat, axes=(-2, -1))
    omega_guess = jnp.fft.irfft2(omega_guess_hat, axes=(-2, -1))

    plot_vort_comp(
        omega_guess[0], omega_trg[0],
        os.path.join(case_dir, "guess_vs_target_t0.svg"),
        l1="Guess vorticity (t0)", l2="Target vorticity (t0)"
    )
    plot_vort_comp(
        omega_guess[-1], omega_trg[-1],
        os.path.join(case_dir, "guess_vs_target_tN.svg"),
        l1="Guess vorticity (tN)", l2="Target vorticity (tN)"
    )
    plot_vort_comp(
        omega_DA[-1], omega_trg[-1],
        os.path.join(case_dir, "DA_vs_target_tN.svg"),
        l1="DA vorticity (tN)", l2="Target vorticity (tN)"
    )
    plot_vort_comp(
        omega_DA[0], omega_trg[0],
        os.path.join(case_dir, "DA_vs_target_t0.svg"),
        l1="DA vorticity (t0)", l2="Target vorticity (t0)"
    )

    plot_field_error_row(
        omega_guess[-1], omega_trg[-1],
        os.path.join(case_dir, "guess_vs_target_tN_error_row.svg"),
        l1="Guess vorticity (tN, pre-opt)", l2="Target vorticity (tN)"
    )
    plot_field_error_row(
        omega_DA[-1], omega_trg[-1],
        os.path.join(case_dir, "DA_vs_target_tN_error_row.svg"),
        l1="DA vorticity (tN, post-opt)", l2="Target vorticity (tN)"
    )

    xp_trg = _load(case_dir, "xp_trg.npy")
    yp_trg = _load(case_dir, "yp_trg.npy")
    xp_DA_obs = _load(case_dir, "xp_DA_obs.npy")
    yp_DA_obs = _load(case_dir, "yp_DA_obs.npy")
    xp_DA_drift = _load(case_dir, "xp_DA_drift.npy")
    yp_DA_drift = _load(case_dir, "yp_DA_drift.npy")
    xp_DA_reset = _load(case_dir, "xp_DA_reset.npy")
    yp_DA_reset = _load(case_dir, "yp_DA_reset.npy")
    t_mask = _load(case_dir, "t_mask.npy")

    plot_particle_tracks(
        xp_trg, yp_trg,
        xp_DA_obs, yp_DA_obs,
        xp_DA_drift, yp_DA_drift,
        xp_DA_reset, yp_DA_reset,
        t_mask,
        os.path.join(case_dir, "particle_tracks.svg")
    )

    vel_error = _load(case_dir, "vel_error.npy")
    time_axis = _load(case_dir, "time_axis.npy")
    plot_vel_error_vs_time(vel_error, time_axis, t_mask, case_dir)

    opt_data = SimpleNamespace(
        loss_record=_load(case_dir, "loss_record.npy"),
        grad_norm_record=_load(case_dir, "grad_norm_record.npy"),
        alpha_gTp_record=_load(case_dir, "alpha_gTp_record.npy"),
        IC_error_record=_load(case_dir, "IC_error_record.npy"),
    )
    plot_convergence(opt_data, case_dir)


def main():
    path = '/Users/noahfrank/Documents/papers/kf-da-proj/figs/Re=100/np=80_nt=4_suc'
    replot_case(path)


if __name__ == "__main__":
    main()
