import jax.numpy as jnp
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.random import key, uniform

import ppptm as gptm

locs = gptm.unit_grid_vars()
nloc = locs.sample_locs.value.shape[0]
y = uniform(key(1), (23, nloc))
y_nan = y.at[0, 0].set(jnp.nan).at[2, 3].set(jnp.nan)


def locwise_param(dist: lsl.Dist, name: str) -> gptm.ParamPredictiveProcessGP:
    param = dist[name]
    assert isinstance(param, gptm.ParamPredictiveProcessGP)
    return param


class TestG:
    def test_init(self):
        gptm.G(y, locs=locs)

    def test_nan_aware_initializers(self):
        gaussian = gptm.G(y_nan, locs=locs).new_gaussian()
        skewnorm = gptm.G(y_nan, locs=locs).new_skewnorm()
        skewt = gptm.G(y_nan, locs=locs).new_skewt()
        gamma = gptm.G(jnp.exp(y_nan), locs=locs).new_gamma()
        weibull = gptm.G(y_nan, locs=locs).new_weibull()

        for g_dist in [gaussian, skewnorm, skewt, gamma, weibull]:
            for param in g_dist.kwinputs.values():
                assert jnp.all(jnp.isfinite(param.value))

    def test_nan_aware_initializers_require_finite_y(self):
        with pytest.raises(ValueError, match="no finite values"):
            gptm.G(jnp.full_like(y, jnp.nan), locs=locs).new_gaussian()

    def test_ard(self):
        g = gptm.G(y, locs=locs, ard=True).new_gaussian()
        assert locwise_param(g, "loc").length_scale.value.size == 2
        assert locwise_param(g, "scale").length_scale.value.size == 2

        g = gptm.G(y, locs=locs, ard=False).new_gaussian()
        assert locwise_param(g, "loc").length_scale.value.size == 1
        assert locwise_param(g, "scale").length_scale.value.size == 1

    def test_gaussian_locwise(self):
        g = gptm.G(y, locs=locs).new_gaussian()
        yvar = lsl.Var.new_obs(y, g).update()
        assert not jnp.any(jnp.isnan(yvar.log_prob))

        assert not jnp.any(jnp.isnan(g["loc"].value))
        assert g["loc"].value.shape == (locs.locs.nloc,)
        assert locwise_param(g, "loc").latent_var.value.shape == (
            locs.locs.ordered_subset.shape[0],
        )

        assert not jnp.any(jnp.isnan(g["scale"].value))
        assert g["scale"].value.shape == (locs.locs.nloc,)
        assert locwise_param(g, "scale").latent_var.value.shape == (
            locs.locs.ordered_subset.shape[0],
        )

    def test_gaussian_const(self):
        g = gptm.G(y, locs=locs).new_gaussian(locwise=[])
        yvar = lsl.Var.new_obs(y, g).update()
        assert not jnp.any(jnp.isnan(yvar.log_prob))

        assert not jnp.any(jnp.isnan(g["loc"].value))
        assert g["loc"].value.shape == ()

        assert not jnp.any(jnp.isnan(g["scale"].value))
        assert g["scale"].value.shape == ()

    def test_skewt_locwise(self):
        g = gptm.G(y, locs=locs).new_skewt()
        yvar = lsl.Var.new_obs(y, g).update()
        assert not jnp.any(jnp.isnan(yvar.log_prob))

        assert not jnp.any(jnp.isnan(g["loc"].value))
        assert g["loc"].value.shape == (locs.locs.nloc,)
        assert locwise_param(g, "loc").latent_var.value.shape == (
            locs.locs.ordered_subset.shape[0],
        )

        assert not jnp.any(jnp.isnan(g["scale"].value))
        assert g["scale"].value.shape == (locs.locs.nloc,)
        assert locwise_param(g, "scale").latent_var.value.shape == (
            locs.locs.ordered_subset.shape[0],
        )

        assert not jnp.any(jnp.isnan(g["skewness"].value))
        assert g["skewness"].value.shape == (locs.locs.nloc,)
        assert locwise_param(g, "skewness").latent_var.value.shape == (
            locs.locs.ordered_subset.shape[0],
        )

        assert not jnp.any(jnp.isnan(g["df"].value))
        assert g["df"].value.shape == ()

    def test_skewt_const(self):
        g = gptm.G(y, locs=locs).new_skewt(locwise=["loc"])
        yvar = lsl.Var.new_obs(y, g).update()
        assert not jnp.any(jnp.isnan(yvar.log_prob))

        assert not jnp.any(jnp.isnan(g["loc"].value))
        assert g["loc"].value.shape == (locs.locs.nloc,)
        assert locwise_param(g, "loc").latent_var.value.shape == (
            locs.locs.ordered_subset.shape[0],
        )

        assert not jnp.any(jnp.isnan(g["scale"].value))
        assert g["scale"].value.shape == ()

        assert not jnp.any(jnp.isnan(g["skewness"].value))
        assert g["skewness"].value.shape == ()

        assert not jnp.any(jnp.isnan(g["df"].value))
        assert g["df"].value.shape == ()

    def test_gamma_locwise(self):
        g = gptm.G(jnp.exp(y), locs=locs).new_gamma()
        yvar = lsl.Var.new_obs(y, g).update()
        assert not jnp.any(jnp.isnan(yvar.log_prob))

        assert not jnp.any(jnp.isnan(g["concentration"].value))
        assert g["concentration"].value.shape == (locs.locs.nloc,)
        assert locwise_param(g, "concentration").latent_var.value.shape == (
            locs.locs.ordered_subset.shape[0],
        )

        assert not jnp.any(jnp.isnan(g["rate"].value))
        assert g["rate"].value.shape == (locs.locs.nloc,)
        assert locwise_param(g, "rate").latent_var.value.shape == (
            locs.locs.ordered_subset.shape[0],
        )

    def test_gamma_const(self):
        g = gptm.G(jnp.exp(y), locs=locs).new_gamma(locwise=[])
        yvar = lsl.Var.new_obs(y, g).update()
        assert not jnp.any(jnp.isnan(yvar.log_prob))

        assert not jnp.any(jnp.isnan(g["rate"].value))
        assert g["rate"].value.shape == ()

        assert not jnp.any(jnp.isnan(g["concentration"].value))
        assert g["concentration"].value.shape == ()


class TestH:
    def test_init(self):
        gptm.H(locs=locs)

    def test_ard(self):
        coef = gptm.H(locs=locs, ard=True).new_coef()
        assert coef.length_scale.value.size == 2

        coef = gptm.H(locs=locs, ard=False).new_coef()
        assert coef.length_scale.value.size == 1

    def test_locwise_amplitude(self):
        coef = gptm.H(locs=locs, locwise_amplitude=True).new_coef()
        assert coef.amplitude.value.size == locs.locs.nloc

        coef = gptm.H(locs=locs, locwise_amplitude=False).new_coef()
        assert coef.amplitude.value.size == 1

    def test_locwise_amplitude_captures_prior(self):
        prior = lsl.Dist(tfd.Weibull, concentration=0.5, scale=0.5)
        h = gptm.H(locs=locs, amplitude_prior=prior)
        amplitude = h.new_amplitude_locwise("amplitude")
        h.amplitude_prior = None

        assert amplitude.mean.dist_node is not None
        assert isinstance(
            amplitude.mean.dist_node.init_dist(), tfd.TransformedDistribution
        )


def _updated(dist: lsl.Dist, name: str):
    node = dist[name]
    lsl.Model([node])
    return node.update().value


def test_locwise_starts_do_not_depend_on_amplitude_start():
    data = 18.5 + 5.0 * y
    dist = gptm.G(data, gptm.unit_grid_vars(), amplitude_start=0.5).new_gaussian()
    assert jnp.allclose(_updated(dist, "loc"), data.mean(), rtol=1e-5)
    assert jnp.allclose(_updated(dist, "scale"), data.std(), rtol=1e-5)


def test_h_locwise_amplitude_starts_at_amplitude_start():
    amplitude = gptm.H(
        gptm.unit_grid_vars(), locwise_amplitude=True, amplitude_start=0.2
    ).new_amplitude("amp")
    lsl.Model([amplitude])
    assert jnp.allclose(amplitude.update().value, 0.2, rtol=1e-5)


@pytest.mark.parametrize("locwise", [[], None])
def test_zero_spread_data_gives_finite_positive_starts(locwise):
    ties = jnp.zeros((10, nloc)).at[0].set(1.0)  # median absolute deviation is 0
    constant = jnp.full((10, nloc), 2.0)  # standard deviation is 0

    def make(data, constructor):
        factory = gptm.G(data, gptm.unit_grid_vars())
        method = getattr(factory, constructor)
        return method() if locwise is None else method(locwise=locwise)

    starts = [
        _updated(make(ties, "new_skewt"), "scale"),
        _updated(make(constant, "new_gaussian"), "scale"),
        _updated(make(constant, "new_gamma"), "concentration"),
        _updated(make(constant, "new_gamma"), "rate"),
    ]
    for start in starts:
        assert jnp.all(jnp.isfinite(start)) and jnp.all(start > 0)
