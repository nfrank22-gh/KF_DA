from kf_da.daComp import KF_Opts, DA_Opts, Particle_Opts, MSE_PP, DA_exp_main
from kf_da.solver.IC_gen import Equilibrium_Init, Gaussian_Init
from kf_da.velInit import AI
from kf_da.opti import ArmijoLineSearch, Joint_Opt, BFGS
from kf_da.icParam import Fourier_Param
import os
from kf_da.utils.create_results_dir import create_results_dir
import yaml
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
def write_hierarchical_case_summary(
    folder,
    filename="results.parquet",
    out_name="case_summary.txt",
):
    """
    Writes a hierarchical text summary:

    T, n_part, NT
        → number of unique case seeds
    """

    folder = Path(folder)
    in_path = folder / filename
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find: {in_path}")

    # --- load file ---
    suffix = in_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_excel(in_path)


    # columns for hierarchy
    top_cols = ["T", "n_part", "NT"]
    required = top_cols + ["case_seed"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # count unique case seeds
    counts = (
        df.groupby(top_cols, dropna=False)
          .agg(n_cases=("case_seed", pd.Series.nunique))
          .reset_index()
          .sort_values(top_cols)
    )

    # write text file
    out_path = folder / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"File: {in_path.name}\n")
        f.write(f"Total rows (after filtering): {len(df)}\n\n")

        for _, row in counts.iterrows():
            f.write(
                f"T={row['T']}, n_part={row['n_part']}, NT={row['NT']}: "
                f"{int(row['n_cases'])} cases\n"
            )

    return out_path

def parquet_to_excel(parquet_path, excel_path=None):
    """
    Copy the contents of a Parquet file to an Excel (.xlsx) file.

    Parameters
    ----------
    parquet_path : str
        Path to the source Parquet file.
    excel_path : str, optional
        Path to the output Excel file. If None, uses same base name.

    Returns
    -------
    str
        Path to the saved Excel file.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Default Excel filename if not provided
    if excel_path is None:
        base = os.path.splitext(parquet_path)[0]
        excel_path = base + ".xlsx"

    # Read parquet and write to Excel
    df = pd.read_parquet(parquet_path)
    df.to_excel(excel_path, index=False)

    print(f"Saved Excel file: {excel_path} (rows={len(df)}, cols={len(df.columns)})")
    return excel_path

def load_config():
    BT_ls = ArmijoLineSearch(alpha_init=1.0, rho=0.25, c=1e-4, max_iters=5)


    #Re = 200 | T = 3.2
    #Re = 100 | T = 3,3
    #Re = 60 | T = 4.1
    #Re = 40 | T = 7.2

    T_dict = {
         200: 3.2,
         100: 3.3,
         60: 4.1,
         40: 7.2
    }

    yaml_root = "../kf-da-configs/daExpConfig.yaml"
    with open(yaml_root) as f:
        daExpConfig = yaml.safe_load(f)
    da_set = daExpConfig["daSet"]
    if da_set["opti"] == "BFGS":
        opti = BFGS(
                ls=BT_ls,
                #Cubic_TR(rho_trg=1, eta_kp=1.0, eta_ki=0, eta_kd=0, eta_min=1e-14, eta_0=1-4, eta_max=1e0),
                #psuedo_proj=Psuedo_Projection(it_list=[24, 49, 74], T=.25),
                its=150, max_mem=20, eps_H=1e-10, print_loss=True)
    elif da_set["sigma_y"] != 0 and da_set["opti"] == "Joint":
        opti =  Joint_Opt(
                state_opt=BFGS(
                ls=BT_ls,
                its=25, max_mem=20, eps_H=1e-10, print_loss=True),
                PP_opt_its=5, opt_loops=6
                )
    sysSet = daExpConfig["sysSet"]
    particle_init_name = sysSet.get("particle_init", "equilibrium")
    if particle_init_name == "equilibrium":
        particle_init = Equilibrium_Init()
    elif particle_init_name == "gaussian":
        particle_init = Gaussian_Init(std=sysSet.get("vel_init_std", 1.0))
    else:
        raise ValueError(
            f"Unknown particle_init: {particle_init_name!r} (expected equilibrium or gaussian)"
        )
    kf_opts = KF_Opts(
        Re = sysSet["Re"],
        n = 4,
        NDOF = sysSet["NDOF"],
        dt = sysSet["dt"],
        total_T=int(float(sysSet["total_T"])),
        min_samp_T=100,
        t_skip=1
    )
    DA_opts = DA_Opts(
        sigma_y=da_set["sigma_y"],
        x__y_sigma=da_set["x__y_sigma"],
        m_dt=da_set["m_dt"],
        sigma_vy=da_set.get("sigma_vy", 0.0),
        vx__vy_sigma=da_set.get("vx__vy_sigma", 1.0),
        n_particles_list=da_set["n_particles_list"],
        NT_list=da_set["NT_list"],
        part_opts=Particle_Opts(St=sysSet["St"], beta=0, particle_init=particle_init),
        n_cases=da_set["n_cases"],
        ic_init=AI(min_norm=.1, max_norm=jnp.inf),
        #ic_init=AI(min_norm=.1, max_norm=.5),
        T_list=[T_dict[kf_opts.Re]],
        optimizer_list=[opti],
        crit_list=[
            MSE_PP(),
            #MSE_Vel()
        ],
        # Preconditioning off by default; set daSet.precond_beta (e.g. 0.1)
        # to re-enable (ADR-0004)
        IC_param_list=[Fourier_Param(kf_opts.NDOF, kf_opts.NDOF//2, beta=da_set.get("precond_beta", 0.0), Re=kf_opts.Re)]
    )

    return DA_opts, kf_opts






def main():
    DA_opts, kf_opts = load_config()

    #Re = 200 | T = 3.2
    #Re = 100 | T = 3,3
    #Re = 60 | T = 4.1
    #Re = 40 | T = 7.2


    case_name = (
            f"DA_Re={kf_opts.Re}_n={kf_opts.n}_dt={kf_opts.dt}_NDOF={kf_opts.NDOF}_mdt={DA_opts.m_dt}"
            f"-St={DA_opts.part_opts.St}_beta={DA_opts.part_opts.beta}_{DA_opts.ic_init}"
            f"{DA_opts.part_opts.particle_init}"
        )

    if DA_opts.sigma_vy > 0:
        noise_dir = (
            f"DA-sigma_y={DA_opts.sigma_y}--x__y_sigma={DA_opts.x__y_sigma}"
            f"--sigma_vy={DA_opts.sigma_vy}--vx__vy_sigma={DA_opts.vx__vy_sigma}"
        )
    elif DA_opts.sigma_y > 0:
        noise_dir = f"DA-sigma_y={DA_opts.sigma_y}--x__y_sigma={DA_opts.x__y_sigma}"
    else:
        noise_dir = "DA-no_noise"
    root = os.path.join(create_results_dir(), noise_dir, case_name)

    DA_exp_main(kf_opts, DA_opts, root)
    parquet_to_excel(os.path.join(root, "results.parquet"), os.path.join(root, "results.xlsx"))
    write_hierarchical_case_summary(root)
    df = pd.read_parquet(os.path.join(root, "results.parquet"))
    df = df.dropna()
    #global_post_main(df, root)


if __name__ == "__main__":
    main()
