import jax.numpy as jnp
import liesel.model as lsl
import pytest
from jax.random import key, normal
from liesel import optim as loptim

import ppptm as gptm

locs = gptm.unit_grid_vars()
nobs = 23
nloc = locs.sample_locs.value.shape[0]
y = normal(key(123), (nobs, nloc))
y_nan = y.at[0, 0].set(jnp.nan).at[2, 3].set(jnp.nan)

fit_locs = gptm.unit_grid_vars(ngrid=4)
fit_nloc = fit_locs.sample_locs.value.shape[0]
fit_y = normal(key(321), (5, fit_nloc))
fit_y_nan = fit_y.at[0, 0].set(jnp.nan).at[2, 3].set(jnp.nan)


def assert_fit_result_is_finite(model, result):
    assert result.best_position
    for value in result.best_position.values():
        assert jnp.all(jnp.isfinite(value))
    assert not jnp.any(jnp.isnan(model.graph.log_prob))


def locwise_param(dist: lsl.Dist, name: str) -> gptm.ParamPredictiveProcessGP:
    param = dist[name]
    assert isinstance(param, gptm.ParamPredictiveProcessGP)
    return param


class TestModel:
    def test_init_gaussian(self):
        model = gptm.Model.new_G(y, locs)

        assert not jnp.any(jnp.isnan(model.response.value))
        assert not jnp.any(jnp.isnan(model.graph.log_prob))

        with pytest.raises(TypeError, match="does not have spatial coefficients"):
            _ = model.spatial_coef

    def test_init_gaussian_mask_nan_response(self):
        model = gptm.Model.new_G(y_nan, locs, mask_nan_response=True)
        mask = jnp.isnan(model.response.value)

        assert jnp.any(mask)
        assert not jnp.any(jnp.isnan(model.graph.log_prob))
        assert jnp.all(model.response.log_prob[mask] == 0.0)

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

    def test_init_hg_mask_nan_response(self):
        model = gptm.Model.new_HG(y_nan, locs, mask_nan_response=True)
        mask = jnp.isnan(model.response.value)

        assert jnp.any(mask)
        assert not jnp.any(jnp.isnan(model.graph.log_prob))
        assert jnp.all(model.response.log_prob[mask] == 0.0)

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

    def test_init_dist_unmasked_with_mask_nan_response(self):
        model = gptm.Model.new_HG(y_nan, locs, mask_nan_response=True)
        dist = model.init_dist()

        assert hasattr(dist, "transformation_and_logdet")
        assert hasattr(dist, "inverse_transformation")

    def test_validation_copy_uses_own_nan_mask(self):
        model = gptm.Model.new_HG(y_nan, locs, mask_nan_response=True)
        validation_y = y.at[1, 1].set(jnp.nan).at[4, 7].set(jnp.nan)

        _, varval = model.graph.copy_nodes_and_vars()
        varval["response"].value = validation_y
        model_validation = lsl.Model([varval["response"]], to_float32=model._to_float32)
        mask = jnp.isnan(validation_y)

        assert not jnp.any(jnp.isnan(model_validation.log_prob))
        assert jnp.all(varval["response"].log_prob[mask] == 0.0)

    def test_h(self):
        model = gptm.Model.new_HG(y, locs)
        samp = model.spatial_coef.latent_var.sample((1,), seed=key(1))[
            "coef_latent"
        ].squeeze()
        model.spatial_coef.latent_var.value = 0.1 * samp

        val = model.h(y)

        assert not jnp.allclose(val, y)
        assert val.shape == y.shape
        assert not jnp.any(jnp.isnan(val))

        vali = model.hi(val)
        assert jnp.allclose(vali, y, atol=1e-4)

    def test_hg(self):
        model = gptm.Model.new_HG(y, locs)
        samp = model.spatial_coef.latent_var.sample((1,), seed=key(1))[
            "coef_latent"
        ].squeeze()
        model.spatial_coef.latent_var.value = 0.1 * samp

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

    def test_log_prob_mask_nan_response(self):
        model = gptm.Model.new_HG(y_nan, locs, mask_nan_response=True)

        val = model.log_prob(y_nan)
        mask = jnp.isnan(y_nan)

        assert val.shape == y_nan.shape
        assert not jnp.any(jnp.isnan(val))
        assert jnp.all(val[mask] == 0.0)

    def test_sample(self):
        model = gptm.Model.new_HG(y, locs)
        samples = model.graph.sample((1,), seed=key(1))
        assert (
            samples["coef_latent"].shape
            == (1,) + model.spatial_coef.latent_var.value.shape
        )
        assert (
            samples["loc_latent"].shape
            == (1,) + locwise_param(model.g_dist, "loc").latent_var.value.shape
        )
        assert (
            samples["scale_latent"].shape
            == (1,) + locwise_param(model.g_dist, "scale").latent_var.value.shape
        )
        assert samples["response"].shape == (1,) + model.response.value.shape

    def test_fit_gaussian_mask_nan_response_location_batching(self):
        model = gptm.Model.new_G(fit_y_nan, fit_locs, mask_nan_response=True)
        result = model.fit(
            stopper=loptim.Stopper(epochs=1, patience=1),
            batch_size=4,
            seed=1,
            save_position_history=False,
        )

        assert jnp.any(jnp.isnan(model.response.value))
        assert_fit_result_is_finite(model, result)

    def test_fit_hg_mask_nan_response_location_batching(self):
        model = gptm.Model.new_HG(
            fit_y_nan, fit_locs, nparam=12, mask_nan_response=True
        )
        result = model.fit(
            stopper=loptim.Stopper(epochs=1, patience=1),
            batch_size=4,
            seed=2,
            save_position_history=False,
        )

        assert jnp.any(jnp.isnan(model.response.value))
        assert_fit_result_is_finite(model, result)

    def test_fit_validation_uses_own_nan_mask_with_location_batching(self):
        validation_y = normal(key(456), (3, fit_nloc))
        validation_y = validation_y.at[1, 2].set(jnp.nan).at[2, 5].set(jnp.nan)
        model = gptm.Model.new_HG(
            fit_y_nan, fit_locs, nparam=12, mask_nan_response=True
        )

        result = model.fit(
            stopper=loptim.Stopper(epochs=1, patience=1),
            response_validation=validation_y,
            batch_size=4,
            seed=3,
            save_position_history=False,
        )

        assert_fit_result_is_finite(model, result)
        assert not jnp.any(jnp.isnan(result.history.loss_validate))

    def test_fit_location_batch_size_must_divide_nloc_without_shuffling(self):
        model = gptm.Model.new_G(fit_y, fit_locs)

        with pytest.raises(ValueError, match="must divide"):
            model.fit(
                stopper=loptim.Stopper(epochs=1, patience=1),
                batch_size=5,
                shuffle_batches=False,
                seed=1,
                save_position_history=False,
            )

    def test_fit_location_batch_size_must_be_integer(self):
        model = gptm.Model.new_G(fit_y, fit_locs)

        with pytest.raises(TypeError, match="positive integer or None"):
            model.fit(batch_size=True)

    def test_fit_location_batch_size_may_leave_remainder_with_shuffling(self):
        model = gptm.Model.new_G(fit_y, fit_locs)

        result = model.fit(
            stopper=loptim.Stopper(epochs=1, patience=1),
            batch_size=5,
            shuffle_batches=True,
            seed=1,
            save_position_history=False,
        )

        assert_fit_result_is_finite(model, result)
