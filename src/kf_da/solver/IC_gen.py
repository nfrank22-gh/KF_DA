import jax
import jax.numpy as jnp
from kf_da.utils.utils import bilinear_sample_periodic

# Particle warmup time (in time units) applied to every DA case; a hidden
# constant rather than a config knob so all experiments share it (ADR-0002).
T_WARMUP = 100.0


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


def _uniform_positions(npart, key, L):
    kx, ky = jax.random.split(key)
    xs = jax.random.uniform(kx, shape=(npart,), minval=0.0, maxval=L)
    ys = jax.random.uniform(ky, shape=(npart,), minval=0.0, maxval=L)
    return xs, ys


def _fluid_velocities(xs, ys, stepper, omega_hat):
    u_hat, v_hat = stepper.NS.vort_hat_2_vel_hat(omega_hat)
    u, v = jnp.fft.irfft2(u_hat), jnp.fft.irfft2(v_hat)
    L = stepper.NS.L
    us = bilinear_sample_periodic(u, xs, ys, L, L)
    vs = bilinear_sample_periodic(v, xs, ys, L, L)
    return us, vs


class Equilibrium_Init:
    """Scatter particles uniform-random on the warm-start snapshot (T_WARMUP
    before the true IC) and co-evolve flow + particles to t=0, so positions
    and velocities sample near the stationary particle distribution. Only the
    particle state is kept; the true IC is always the stored snapshot."""

    def make_particle_IC(self, npart, key, stepper, omega_warm_start_hat):
        xp, yp = _uniform_positions(npart, key, stepper.NS.L)
        up, vp = _fluid_velocities(xp, yp, stepper, omega_warm_start_hat)
        nsteps = int(round(T_WARMUP / stepper.dt))

        def body(carry, _):
            return stepper(*carry), None

        (_, xp, yp, up, vp), _ = jax.lax.scan(
            body, (omega_warm_start_hat, xp, yp, up, vp), xs=None, length=nsteps
        )
        return xp, yp, up, vp

    def __repr__(self):
        return "-pinit=eq"


class Gaussian_Init:
    """Uniform-random positions + i.i.d. N(0, std^2) velocities placed
    directly at t=0 (no particle warmup); requires inertial particles."""

    def __init__(self, std=1.0):
        self.std = std

    def make_particle_IC(self, npart, key, stepper, omega_warm_start_hat):
        pos_key, ku, kv = jax.random.split(key, 3)
        xp, yp = _uniform_positions(npart, pos_key, stepper.NS.L)
        up = self.std * jax.random.normal(ku, (npart,))
        vp = self.std * jax.random.normal(kv, (npart,))
        return xp, yp, up, vp

    def __repr__(self):
        return f"-pinit=gauss-std={self.std}"
