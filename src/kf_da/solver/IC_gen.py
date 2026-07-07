import numpy as np
import jax.numpy as jnp
from kf_da.utils.utils import bilinear_sample_periodic, Specteral_Upsampling


import jax
import jax.numpy as jnp

def init_particles_vector(
    n: int,
    u: jnp.ndarray,
    v: jnp.ndarray,
    x_range,
    y_range,
    L: float,
    seed: int = 0,
):


    key = jax.random.PRNGKey(seed)
    kx, ky = jax.random.split(key, 2)
    xs = jax.random.uniform(kx, shape=(n,), minval=x_range[0], maxval=x_range[1])
    ys = jax.random.uniform(ky, shape=(n,), minval=y_range[0], maxval=y_range[1])

    us = bilinear_sample_periodic(u, xs, ys, L, L)
    vs = bilinear_sample_periodic(v, xs, ys, L, L)

    return xs, ys, us, vs


def _fluid_particle_IC(npart, seed, stepper, omega_hat):
    u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega_hat)
    u, v = jnp.fft.irfft2(u_hat), jnp.fft.irfft2(v_hat)
    L = stepper.NS.L
    return init_particles_vector(npart, u, v, (0, L), (0, L), L, seed=seed)


class Fluid_Vel_Init:
    """Velocity = fluid velocity at the particle position (current default)."""

    def make_particle_IC(self, npart, seed, stepper, omega0_hat, warmup_ctx=None):
        return _fluid_particle_IC(npart, seed, stepper, omega0_hat)

    def __repr__(self):
        return ""


class Gaussian_Vel_Init:
    """Velocity drawn i.i.d. from N(0, std^2); positions as in Fluid_Vel_Init."""

    def __init__(self, std=1.0):
        self.std = std

    def make_particle_IC(self, npart, seed, stepper, omega0_hat, warmup_ctx=None):
        xp, yp, _, _ = _fluid_particle_IC(npart, seed, stepper, omega0_hat)
        # fold_in(..., 2): the observation-noise stream uses fold_in(..., 1)
        key = jax.random.fold_in(jax.random.PRNGKey(seed), 2)
        ku, kv = jax.random.split(key)
        up = self.std * jax.random.normal(ku, (npart,))
        vp = self.std * jax.random.normal(kv, (npart,))
        return xp, yp, up, vp

    def __repr__(self):
        return f"-vinit=gauss-std={self.std}"


class Warmup_Vel_Init:
    """Seed particles on the attractor snapshot T_w before omega0_hat and
    co-evolve flow + particles for T_w; the final state is the particle IC."""

    def __init__(self, T_w):
        self.T_w = T_w

    def snapshot_offset(self, t_skip):
        k = self.T_w / t_skip
        if self.T_w <= 0 or abs(k - round(k)) > 1e-9:
            raise ValueError(
                f"Warmup T_w={self.T_w} must be a positive multiple of t_skip={t_skip}."
            )
        return int(round(k))

    def make_particle_IC(self, npart, seed, stepper, omega0_hat, warmup_ctx=None):
        snapshots, idx, t_skip = warmup_ctx
        k = self.snapshot_offset(t_skip)
        if idx - k < 0:
            raise ValueError(
                f"Snapshot idx={idx} is too early for warmup T_w={self.T_w} "
                f"(needs idx >= {k}); drop this seed or reduce T_w."
            )
        omega_start = jnp.asarray(snapshots[idx - k])

        xp, yp, up, vp = _fluid_particle_IC(npart, seed, stepper, omega_start)
        nsteps = int(round(self.T_w / stepper.dt))

        def body(carry, _):
            return stepper(*carry), None

        (_, xp, yp, up, vp), _ = jax.lax.scan(
            body, (omega_start, xp, yp, up, vp), xs=None, length=nsteps
        )
        return xp, yp, up, vp

    def __repr__(self):
        return f"-vinit=warmup-Tw={self.T_w}"
