import jax.numpy as jnp
from jax.random import key, normal

import ppptm as gptm

locs = gptm.unit_grid_vars()
nobs = 23

y = normal(key(123), (nobs, locs.locs.nloc))


class TestModel:
    def test_init_gaussian(self):
        model = gptm.Model.new_G(y, locs)

        assert not jnp.any(jnp.isnan(model.response.value))
        assert not jnp.any(jnp.isnan(model.graph.log_prob))

    def test_init_gaussian_const(self):
        g = gptm.G(y, locs).new_gaussian(locwise=[])
        model = gptm.Model.new_G(y, locs, g_dist=g)

        assert not jnp.any(jnp.isnan(model.response.value))
        assert not jnp.any(jnp.isnan(model.graph.log_prob))

    def test_init_skewt(self):
        g = gptm.G(y, locs).new_skewt()
        model = gptm.Model.new_G(y, locs, g_dist=g)

        assert not jnp.any(jnp.isnan(model.response.value))
        assert not jnp.any(jnp.isnan(model.graph.log_prob))

    def test_init_hg(self):
        model = gptm.Model.new_HG(y, locs)
        assert not jnp.any(jnp.isnan(model.response.value))
        assert not jnp.any(jnp.isnan(model.graph.log_prob))

    def test_init_hg_ard(self):
        coef = gptm.H(locs, ard=True).new_coef()
        model = gptm.Model.new_HG(y, locs, coef=coef)
        assert not jnp.any(jnp.isnan(model.response.value))
        assert not jnp.any(jnp.isnan(model.graph.log_prob))

    def test_init_hg_locwise_amp(self):
        coef = gptm.H(locs, ard=True, locwise_amplitude=True).new_coef()
        model = gptm.Model.new_HG(y, locs, coef=coef)
        assert not jnp.any(jnp.isnan(model.response.value))
        assert not jnp.any(jnp.isnan(model.graph.log_prob))

    def test_parameter_names(self):
        model = gptm.Model.new_HG(y, locs)
        names = [
            "scale_amplitude_transformed",
            "scale_mean",
            "scale_length_scale_transformed",
            "scale_latent",
            "loc_amplitude_transformed",
            "loc_mean",
            "loc_length_scale_transformed",
            "loc_latent",
            "coef_amplitude_transformed",
            "coef_length_scale_transformed",
            "coef_latent",
        ]
        for name in names:
            assert name in model.parameters

        for name in model.parameters:
            assert name in names

    def test_init_dist(self):
        model = gptm.Model.new_HG(y, locs)
        dist = model.init_dist()
        assert dist is not None

    def test_h(self):
        model = gptm.Model.new_HG(y, locs)
        samp = model.coef.latent_var.sample((1,), seed=key(1))["coef_latent"].squeeze()
        model.coef.latent_var.value = 0.1 * samp

        val = model.h(y)

        assert not jnp.allclose(val, y)
        assert val.shape == y.shape
        assert not jnp.any(jnp.isnan(val))

        vali = model.hi(val)
        assert jnp.allclose(vali, y, atol=1e-4)

    def test_hg(self):
        model = gptm.Model.new_HG(y, locs)
        samp = model.coef.latent_var.sample((1,), seed=key(1))["coef_latent"].squeeze()
        model.coef.latent_var.value = 0.1 * samp

        val = model.hg(y)

        assert not jnp.allclose(val, y)
        assert val.shape == y.shape
        assert not jnp.any(jnp.isnan(val))

        vali = model.hgi(val)
        assert jnp.allclose(vali, y, atol=1e-4)

    def test_g(self):
        model = gptm.Model.new_HG(y, locs)

        val = model.g(y)

        assert not jnp.allclose(val, y)
        assert val.shape == y.shape
        assert not jnp.any(jnp.isnan(val))

        vali = model.gi(val)
        assert jnp.allclose(vali, y, atol=1e-4)

    def test_log_prob(self):
        model = gptm.Model.new_HG(y, locs)

        val = model.log_prob(y)

        assert not jnp.allclose(val, y)
        assert val.shape == y.shape
        assert not jnp.any(jnp.isnan(val))

    def test_sample(self):
        model = gptm.Model.new_HG(y, locs)
        samples = model.graph.sample((1,), seed=key(1))
        assert samples["coef_latent"].shape == (1,) + model.coef.latent_var.value.shape
        assert (
            samples["loc_latent"].shape
            == (1,) + model.g_dist["loc"].latent_var.value.shape
        )
        assert (
            samples["scale_latent"].shape
            == (1,) + model.g_dist["scale"].latent_var.value.shape
        )
        assert samples["response"].shape == (1,) + model.response.value.shape
