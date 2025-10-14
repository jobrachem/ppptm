from __future__ import annotations

from copy import deepcopy

import jax.numpy as jnp
import liesel.model as lsl
import pandas as pd
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax.random import key
from jax.scipy.linalg import solve_triangular
from jax.typing import ArrayLike

from ..util.locs import LocationVars
from .kernel import GPKernel


def brownian_motion_mat(nrows: int, ncols: int):
    r = jnp.arange(nrows)[:, None] + 1
    c = jnp.arange(ncols)[None, :] + 1
    return jnp.minimum(r, c)


def rw_weight_matrix(D: int):
    C = jnp.eye(D - 1) - jnp.ones(D - 1) / (D - 1)
    B = brownian_motion_mat(D - 2, D - 2)
    L = jnp.linalg.cholesky(B, upper=False)
    W = C @ jnp.r_[jnp.zeros((1, D - 2)), L]
    return W


class RandomWalkParamPredictiveProcessGP(lsl.Var):
    def __init__(
        self,
        gp_kernel: GPKernel,
        locs: LocationVars,
        nparam: int,
        bijector: tfb.Bijector = tfb.Identity(),
        name: str = "",
        salt: float = 1e-6,
    ) -> None:
        K = gp_kernel
        D = nparam
        Kuu = K.new(
            x1=locs.ordered_subset,
            x2=locs.ordered_subset,
            amplitude=jnp.array(1.0),
            name=f"{name}_Kuu",
        ).update()

        Kdu = K.new(
            x1=locs.ordered,
            x2=locs.ordered_subset,
            amplitude=jnp.array(1.0),
            name=f"{name}_Kdu",
        ).update()

        W = brownian_motion_mat(D, D)

        n_inducing_locs = Kuu.value.shape[0]
        M = n_inducing_locs

        self.latent_var = lsl.Var.new_param(
            jnp.zeros((n_inducing_locs * (W.shape[1]),)),
            distribution=lsl.Dist(tfd.Normal, loc=jnp.array(0.0), scale=jnp.array(1.0)),
            name=f"{name}_latent",
        )

        def _compute_param(latent_var, Kuu, Kdu, amplitude):
            """
            latent_var: (M*D,)
            Kuu: (M, M)
            Kdu: (N, M)

            Returns: (N,)
            """
            Kuu = Kuu.at[jnp.diag_indices(M)].add(salt)
            L = jnp.linalg.cholesky(Kuu)  # (M x M)

            Bt = jnp.reshape(latent_var, shape=(n_inducing_locs, W.shape[1]))
            B = solve_triangular(L, Bt, trans="T", lower=True)  # (M, D)
            delta_mat = (W @ B.T) @ Kdu.T  # (D, D) @ (D, M) @ (M, N) -> (D, N)
            delta_mat = delta_mat * jnp.expand_dims(amplitude, 0)
            return delta_mat.T  # (N, D)

        super().__init__(
            lsl.Calc(
                _compute_param,
                self.latent_var,
                Kuu,
                Kdu,
                K.amplitude,
            ),
            name=name,
        )

        self.bijector = bijector
        self.amplitude = K.amplitude
        self.length_scale = K.length_scale
        self.kernel = K.kernel
        self.Kuu = Kuu
        self.Kdu = Kdu
        self.locs = locs
        self.parameter_names = [self.latent_var.name]

    @staticmethod
    def sample_df(
        locs: LocationVars,
        nparam: int = 5,
        n: int = 1,
        seed: int = 1,
        salt: float = 1e-6,
        amplitude: ArrayLike = 0.1,
        length_scale: ArrayLike = 0.1,
    ) -> pd.DataFrame:
        amplitude = jnp.asarray(amplitude)
        length_scale = jnp.asarray(length_scale)

        K = GPKernel(
            tfk.ExponentiatedQuadratic, amplitude=amplitude, length_scale=length_scale
        )
        var = RandomWalkParamPredictiveProcessGP(
            locs=deepcopy(locs),
            nparam=nparam,
            gp_kernel=K,
            name="v",
            salt=salt,
        )

        model = lsl.Model([var], to_float32=False)
        samples = model.sample(seed=key(seed), shape=(n, 1))
        var_samples = var.predict(samples).squeeze(1)

        n, nloc, D = var_samples.shape
        var_samples_long = jnp.reshape(var_samples, (n, nloc * D))
        df = pd.DataFrame(var_samples_long.T)
        df["D"] = jnp.tile(jnp.arange(D), nloc)
        df["loc"] = jnp.repeat(jnp.arange(nloc), D)
        df["lon"] = locs.ordered.value[df["loc"].to_numpy()][:, 0]
        df["lat"] = locs.ordered.value[df["loc"].to_numpy()][:, 1]
        df.melt(id_vars=["D", "loc", "lon", "lat"], var_name="sample")

        return df.melt(id_vars=["lon", "lat", "loc", "D"], var_name="sample")

    @staticmethod
    def sample_array(
        locs: LocationVars,
        nparam: int = 5,
        n: int = 1,
        seed: int = 1,
        salt: float = 1e-6,
        amplitude: ArrayLike = 0.1,
        length_scale: ArrayLike = 0.1,
    ) -> pd.DataFrame:
        amplitude = jnp.asarray(amplitude)
        length_scale = jnp.asarray(length_scale)

        K = GPKernel(
            tfk.ExponentiatedQuadratic, amplitude=amplitude, length_scale=length_scale
        )
        var = RandomWalkParamPredictiveProcessGP(
            locs=deepcopy(locs),
            nparam=nparam,
            gp_kernel=K,
            name="v",
            salt=salt,
        )

        model = lsl.Model([var], to_float32=False)
        samples = model.sample(seed=key(seed), shape=(n, 1))
        var_samples = var.predict(samples).squeeze(1)
        return var_samples


SpatPTMCoef = RandomWalkParamPredictiveProcessGP
GPTMCoef = RandomWalkParamPredictiveProcessGP
