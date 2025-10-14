import jax.numpy as jnp
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax.random import key, normal

import ppptm as gptm

locs = gptm.unit_grid_vars()


class TestPPParam:
    def test_init(self):
        var = gptm.ParamPredictiveProcessGP(
            locs=locs,
            gp_kernel=gptm.GPKernel(tfk.ExponentiatedQuadratic),
        )

        var.latent_var.value = normal(key(2), (locs.ordered_subset.value.shape[0],))
        var.update()
        assert not jnp.any(jnp.isnan(var.value))

    def test_sample_df(self):
        df = gptm.GPVar.sample_df(locs)
        assert df.shape == (100, 5)

        df = gptm.GPVar.sample_df(locs, n=2)
        assert df.shape == (200, 5)

    def test_sample_array(self):
        arr = gptm.GPVar.sample_array(locs)
        assert arr.shape == (1, 100)

        arr = gptm.GPVar.sample_array(locs, n=3)
        assert arr.shape == (3, 100)
