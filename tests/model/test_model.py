import jax
import jax.numpy as jnp
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.random import key, normal
from liesel import optim as loptim

import ppptm as gptm

EMA = loptim.EmaTrainLossMonitor(effective_window=1.0)

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
    assert result.position_min_monitor
    for value in result.position_min_monitor.values():
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
        assert jnp.all(jnp.asarray(model.response.log_prob)[mask] == 0.0)

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
        assert jnp.all(jnp.asarray(model.response.log_prob)[mask] == 0.0)

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
            loss_monitor=EMA,
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
            loss_monitor=EMA,
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
            loss_monitor="validation",
            stopper=loptim.Stopper(epochs=1, patience=1),
            response_validation=validation_y,
            batch_size=4,
            seed=3,
            save_position_history=False,
        )

        assert_fit_result_is_finite(model, result)
        assert result.monitor_source == "validation"
        assert not jnp.any(jnp.isnan(result.history.loss_monitor))

    def test_fit_location_batch_size_must_divide_nloc_without_shuffling(self):
        model = gptm.Model.new_G(fit_y, fit_locs)

        with pytest.raises(ValueError, match="must divide"):
            model.fit(
                loss_monitor=EMA,
                stopper=loptim.Stopper(epochs=1, patience=1),
                batch_size=5,
                shuffle_batches=False,
                seed=1,
                save_position_history=False,
            )

    def test_fit_location_batch_size_must_be_integer(self):
        model = gptm.Model.new_G(fit_y, fit_locs)

        with pytest.raises(TypeError, match="positive integer or None"):
            model.fit(loss_monitor=EMA, batch_size=True)

    def test_fit_location_batch_size_may_leave_remainder_with_shuffling(self):
        model = gptm.Model.new_G(fit_y, fit_locs)

        result = model.fit(
            loss_monitor=EMA,
            stopper=loptim.Stopper(epochs=1, patience=1),
            batch_size=5,
            shuffle_batches=True,
            seed=1,
            save_position_history=False,
        )

        assert_fit_result_is_finite(model, result)

    def test_fit_validation_without_auto_update(self):
        # Without auto_update the prior log-probs start uncomputed; liesel's
        # validation monitor needs an up-to-date model state.
        model = gptm.Model.new_HG(fit_y, fit_locs, nparam=12, auto_update=False)

        result = model.fit(
            loss_monitor="validation",
            response_validation=normal(key(789), (3, fit_nloc)),
            stopper=loptim.Stopper(epochs=1, patience=1),
            save_position_history=False,
        )

        assert_fit_result_is_finite(model, result)

    def test_fit_requires_explicit_loss_monitor(self):
        model = gptm.Model.new_G(fit_y, fit_locs)

        with pytest.raises(TypeError, match="loss_monitor"):
            model.fit(stopper=loptim.Stopper(epochs=1, patience=1))  # ty: ignore[missing-argument]

    def test_fit_passes_loss_monitor_to_liesel(self):
        model = gptm.Model.new_G(fit_y, fit_locs)

        result = model.fit(
            loss_monitor="train_full_data",
            stopper=loptim.Stopper(epochs=1, patience=1),
            save_position_history=False,
        )

        assert result.monitor_source == "train_full_data"


def test_coefficient_gp_likelihood_gradients():
    locations = gptm.unit_grid_vars(ngrid=2, n_subset=3)
    coef = gptm.H(locations, nparam=8).new_coef()
    response = jnp.linspace(-2.0, 2.5, 20).reshape(5, 4)
    g_dist = lsl.Dist(tfd.TwoPieceStudentT, df=8.0, loc=0.0, scale=1.0, skewness=1.3)
    model = gptm.Model.new_HG(
        response, locations, nparam=8, coef=coef, g_dist=g_dist, locscale=False
    )
    position = {name: var.value for name, var in model.graph.parameters.items()}
    position["coef_latent"] = jnp.linspace(-0.3, 0.4, 24)
    assert model.spatial_coef.value.shape == (4, 8)

    def log_likelihood(params):
        return model.log_prob(response, samples=params).sum()

    value, gradients = jax.jit(jax.value_and_grad(log_likelihood))(position)
    assert jnp.isfinite(value)
    assert jnp.allclose(value, log_likelihood(position), rtol=1e-5)
    for name in (
        "coef_latent",
        "coef_amplitude_transformed",
        "coef_length_scale_transformed",
    ):
        assert jnp.all(jnp.isfinite(gradients[name]))
        assert jnp.linalg.norm(gradients[name]) > 1e-6

    for data in (response, response[:2]):
        log_prob = model.log_prob(data, samples=position)
        assert log_prob.shape == data.shape
        assert jnp.all(jnp.isfinite(log_prob))
        state = model.graph.update_state(position | {"response": data})
        assert jnp.allclose(log_prob.sum(), state["_model_log_lik"].value)

    grid = jnp.broadcast_to(jnp.linspace(-5.0, 5.0, 101)[:, None], (101, 4))
    transformed = jax.jit(lambda p: model.hg(grid, samples=p))(position)
    assert transformed.shape == grid.shape
    assert jnp.all(jnp.isfinite(transformed))
    assert jnp.all(jnp.diff(transformed, axis=0) > 0.0)


def test_second_model_on_same_locs_explains_how_to_copy():
    shared = gptm.unit_grid_vars(ngrid=3)
    response = normal(key(11), (4, 9))
    baseline = gptm.Model.new_G(response, shared)

    with pytest.raises(ValueError, match="copy"):
        gptm.Model.new_HG(response, shared, nparam=6)

    flexible = gptm.Model.new_HG(response, shared.copy(), nparam=6)
    assert baseline.graph is not flexible.graph


def test_unbatched_position_predicts_at_new_locations():
    response = normal(key(12), (4, 9))
    model = gptm.Model.new_HG(response, gptm.unit_grid_vars(ngrid=3), nparam=6)
    position = {name: model.graph.vars[name].value for name in model.parameters}
    new_locs = jnp.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.1], [0.3, 0.8], [0.7, 0.7]])
    new_response = normal(key(13), (3, 5))
    newdata = {"sample_locs": new_locs}

    unbatched = model.hg(new_response, samples=position, newdata=newdata)
    batched = model.hg(
        new_response,
        samples={name: value[None] for name, value in position.items()},
        newdata=newdata,
    )
    assert unbatched.shape == (3, 5)
    assert jnp.allclose(unbatched, batched[0], atol=1e-5)


def test_scale_loss_divides_by_the_number_of_training_fields():
    def first_loss(scale_loss: bool) -> float:
        model = gptm.Model.new_HG(fit_y, fit_locs.copy(), nparam=6)
        result = model.fit(
            loss_monitor=EMA,
            stopper=loptim.Stopper(epochs=1, patience=1),
            scale_loss=scale_loss,
            save_position_history=False,
        )
        return float(result.history.loss_train[0])

    nobs = fit_y.shape[0]
    assert first_loss(False) / first_loss(True) == pytest.approx(nobs, rel=1e-4)


def test_response_columns_must_match_locations():
    with pytest.raises(ValueError, match="locations"):
        gptm.Model.new_G(jnp.ones((5, 1)), gptm.unit_grid_vars(ngrid=2))


def test_fit_is_reproducible_by_default():
    import inspect

    assert inspect.signature(gptm.Model.fit).parameters["seed"].default == 0
