import jax.numpy as jnp
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax.random import key, normal

import ppptm as gptm

locs = gptm.unit_grid_vars()


class TestPPParam:
    def test_init(self):
        D = 10
        var = gptm.RandomWalkParamPredictiveProcessGP(
            locs=locs,
            nparam=D,
            gp_kernel=gptm.GPKernel(tfk.ExponentiatedQuadratic),
        )

        var.latent_var.value = normal(key(2), (locs.ordered_subset.value.shape[0] * D,))
        var.update()
        assert not jnp.any(jnp.isnan(var.value))

    def test_sample_df(self):
        df = gptm.GPTMCoef.sample_df(locs)
        assert df.shape == (500, 6)

        df = gptm.GPTMCoef.sample_df(locs, n=2)
        assert df.shape == (1000, 6)

    def test_sample_array(self):
        arr = gptm.GPTMCoef.sample_array(locs)
        assert arr.shape == (1, 100, 5)

        arr = gptm.GPTMCoef.sample_array(locs, n=3)
        assert arr.shape == (3, 100, 5)
