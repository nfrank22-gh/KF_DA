from dataclasses import dataclass, field
from kf_da.velInit.IC_init import IC_init
from kf_da.solver.IC_gen import Equilibrium_Init

@dataclass
class KF_Opts:
    Re: float
    n: int
    NDOF: int
    dt: float
    total_T: float
    min_samp_T: float
    t_skip: float

@dataclass
class Particle_Opts:
    St: float
    beta: float
    particle_init: any = field(default_factory=Equilibrium_Init)

@dataclass
class DA_Opts:
    sigma_y: float
    x__y_sigma: float
    m_dt: any
    n_particles_list: any
    NT_list: any
    part_opts: Particle_Opts
    n_cases: int
    ic_init: IC_init
    optimizer_list: any
    crit_list: any
    IC_param_list: any
    T_list: any
    sigma_vy: float = 0.0
    vx__vy_sigma: float = 1.0
