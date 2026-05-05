import jax.numpy as jnp


class Fourier_Param:
    def __init__(self, Nx: int, K: int, beta, Re):
        self.Nx = int(Nx)
        self.K = int(K)
        self.beta = float(beta)

        if self.K < 0:
            raise ValueError("K must be nonnegative.")
        if self.K > self.Nx // 2:
            raise ValueError(f"K={self.K} must be <= Nx//2={self.Nx//2}.")

        # rfft2 spectrum shape for an (Nx, Nx) real field
        self.full_shape = (self.Nx, self.Nx // 2 + 1)

        # kx is stored only for nonnegative modes in rfft2
        self.kx_idx = jnp.arange(self.K + 1)

        # ky uses wrap-around indexing to represent [-K..K]
        if (self.Nx % 2 == 0) and (self.K == self.Nx // 2):
            self.ky_idx = jnp.arange(self.Nx)
        else:
            ky_pos = jnp.arange(self.K + 1)                  # 0..K
            ky_neg = jnp.arange(self.Nx - self.K, self.Nx)   # Nx-K..Nx-1
            self.ky_idx = jnp.concatenate([ky_pos, ky_neg], axis=0)

        self.nky = int(self.ky_idx.shape[0])
        self.nkx = int(self.kx_idx.shape[0])
        self.small_shape = (self.nky, self.nkx)

        self.nn = self.nky * self.nkx
        self.out_dim = 2 * self.nn

        # cached broadcasted indices
        self._KY = self.ky_idx[:, None]
        self._KX = self.kx_idx[None, :]

        # signed integer wavenumbers
        self.kx_vals = self.kx_idx
        self.ky_vals = jnp.where(self.ky_idx <= self.Nx // 2,
                                 self.ky_idx,
                                 self.ky_idx - self.Nx)

        # k^2 = kx^2 + ky^2 on the retained block
        kx_grid = self.kx_vals[None, :]   # (1, nkx)
        ky_grid = self.ky_vals[:, None]   # (nky, 1)
        self.k2 = kx_grid**2 + ky_grid**2

        # P(k) = exp(-½ ν β k²) mimics diffusion in spectral space.
        # Optimizing in the whitened variable s = P⁻¹ω̂ gives modes roughly
        # unit magnitude, improving the conditioning of the optimization problem.
        self.precond = jnp.exp(-0.5 * (1/Re) * self.beta * self.k2)
        self.precond_inv = jnp.exp(0.5 * (1/Re) * self.beta * self.k2)

    @staticmethod
    def _real_to_complex_dtype(dtype):
        if dtype == jnp.float32:
            return jnp.complex64
        elif dtype == jnp.float64:
            return jnp.complex128
        raise TypeError(f"Unsupported dtype: {dtype}")

    def _z_to_full(self, z: jnp.ndarray, scale=None) -> jnp.ndarray:
        """Unpack real vector [re; im] into full rfft2 spectrum, optionally scaling modes."""
        c_dtype = self._real_to_complex_dtype(z.dtype)
        re = z[:self.nn].reshape(self.small_shape)
        im = z[self.nn:].reshape(self.small_shape)
        U_small = re + 1j * im
        if scale is not None:
            U_small = scale * U_small
        omega_hat_full = jnp.zeros(self.full_shape, dtype=c_dtype)
        return omega_hat_full.at[self._KY, self._KX].set(U_small.astype(c_dtype))

    def pack(self, omega_hat: jnp.ndarray) -> jnp.ndarray:
        """
        Extract retained Fourier block and pack into a real vector.
        omega_hat: complex array, shape (Nx, Nx//2+1)
        returns: real array, shape (2*nky*nkx,)
        """
        if omega_hat.shape != self.full_shape:
            raise ValueError(
                f"Expected omega_hat shape {self.full_shape}, got {omega_hat.shape}"
            )

        U_small = omega_hat[self._KY, self._KX]
        flat = U_small.reshape(-1)
        return jnp.concatenate([flat.real, flat.imag], axis=0)

    def unpack(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Unpack real vector into truncated Fourier block and scatter into full rfft2 array.
        z: real array, shape (2*nky*nkx,)
        returns: complex rfft2 spectrum, shape (Nx, Nx//2+1)
        """
        z = jnp.asarray(z)
        if z.shape != (self.out_dim,):
            raise ValueError(f"Expected z shape ({self.out_dim},), got {z.shape}")
        return self._z_to_full(z)

    def transform(self, omega_hat: jnp.ndarray) -> jnp.ndarray:
        """
        Map omega_hat -> s-space packed vector.

        Since omega_hat = P * s_hat,
        we have s_hat = P^{-1} * omega_hat.
        """
        if omega_hat.shape != self.full_shape:
            raise ValueError(
                f"Expected omega_hat shape {self.full_shape}, got {omega_hat.shape}"
            )

        U_small = omega_hat[self._KY, self._KX]
        S_small = self.precond_inv * U_small
        flat = S_small.reshape(-1)
        Z = jnp.concatenate([flat.real, flat.imag], axis=0)
        return Z

    def inv_transform(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Map packed s-space vector -> omega_hat.

        Since omega_hat = P * s_hat.
        """
        z = jnp.asarray(z)
        if z.shape != (self.out_dim,):
            raise ValueError(f"Expected z shape ({self.out_dim},), got {z.shape}")
        return self._z_to_full(z, scale=self.precond)

    def __repr__(self) -> str:
        return f"Fourier_K={self.K}, beta={self.beta}"