# import jax
# import jax.numpy as jnp
# import jax.random as jrd
# import pytest
# import tensorflow_probability.substrates.jax.distributions as tfd
# from liesel_ptm import sample_shape
# from liesel_ptm.ptm import OnionCoef, OnionKnots
# from liesel_ptm.ptm.util import sample_simple_ptm
# from tensorflow_probability.substrates.jax import tf2jax as tf

# from ppptm.dist import (
#     CensoredTransformationDist,
#     CustomGEV,
#     LocScaleCensoredTransformationDist,
# )


# @pytest.mark.parametrize(
#     "loc, scale, concentration, values",
#     [
#         # Test GEV with positive concentration
#         (0.0, 1.0, 0.5, [0.5, 1.0, 1.5]),
#         # Test GEV with negative concentration (upper bound support)
#         (0.0, 1.0, -0.5, [-1.0, -0.5, 0.0]),
#     ],
# )
# def test_custom_gev_log_prob(loc, scale, concentration, values):
#     """Test log_prob for CustomGEV for different parameters and values."""
#     gev = CustomGEV(loc=loc, scale=scale, concentration=concentration)
#     gev_classic = tfd.GeneralizedExtremeValue(
#         loc=loc, scale=scale, concentration=concentration
#     )

#     values_tf = tf.constant(values, dtype=tf.float32)

#     log_probs = gev.log_prob(values_tf)
#     log_probs_classic = gev_classic.log_prob(values_tf)

#     assert jnp.allclose(log_probs, log_probs_classic)


# def test_custom_gev_gumbel_fallback():
#     """Test that CustomGEV falls back to Gumbel when concentration == 0."""
#     loc = 0.0
#     scale = 1.0
#     concentration = 0.0

#     gev = CustomGEV(loc=loc, scale=scale, concentration=concentration)
#     gumbel = tfd.Gumbel(loc=loc, scale=scale)

#     values = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)

#     gev_log_probs = gev.log_prob(values)
#     gumbel_log_probs = gumbel.log_prob(values)

#     # Assert that CustomGEV returns the same log_prob as Gumbel when concentration == 0
#     tf.debugging.assert_near(gev_log_probs, gumbel_log_probs, atol=1e-5)


# def test_custom_gev_gumbel_fallback_shape():
#     loc = jnp.zeros(100)
#     scale = jnp.ones(100)
#     concentration = jnp.zeros(100)

#     gev = CustomGEV(loc=loc, scale=scale, concentration=concentration)
#     gumbel = tfd.Gumbel(loc=loc, scale=scale)

#     key = jrd.PRNGKey(42)
#     values = jrd.gumbel(key, shape=(30, 100))

#     # Compute log probabilities using both CustomGEV and Gumbel
#     gev_log_probs = gev.log_prob(values)
#     gumbel_log_probs = gumbel.log_prob(values)

#     assert gev_log_probs.shape == (30, 100)
#     assert gumbel_log_probs.shape == (30, 100)

#     assert jnp.allclose(gev_log_probs, gumbel_log_probs)


# @pytest.mark.parametrize(
#     "loc, scale, concentration, value",
#     [
#         (0.0, 1.0, 0.5, -2.0),  # Outside upper bound (concentration > 0)
#         (0.0, 1.0, -0.5, 2.0),  # Outside lower bound (concentration < 0)
#     ],
# )
# def test_outside_support(loc, scale, concentration, value):
#     """Test that log_prob returns a large negative value for values outside the support."""
#     # Initialize the CustomGEV distribution
#     gev = CustomGEV(loc=loc, scale=scale, concentration=concentration)

#     # Test value outside support
#     value_tf = tf.constant([value], dtype=tf.float32)
#     log_prob = gev.log_prob(value_tf)

#     # Assert that the log_prob for values outside support is a large negative value
#     assert log_prob < -800


# def test_grad():
#     loc = 0.0
#     scale = 1.0
#     concentration = 0.5

#     x = jnp.linspace(-5, 5, 50)

#     def log_prob_fn(x, position):
#         gev = CustomGEV(
#             loc=position["loc"],
#             scale=position["scale"],
#             concentration=position["concentration"],
#         )
#         lp = gev.log_prob(x).sum()
#         return lp

#     lp = log_prob_fn(x, dict(loc=loc, scale=scale, concentration=concentration))

#     assert jnp.all(~jnp.isnan(lp))
#     assert jnp.all(~jnp.isinf(lp))

#     lp_grad_fn = jax.grad(log_prob_fn, argnums=1)
#     lp_grad = lp_grad_fn(x, dict(loc=loc, scale=scale, concentration=concentration))

#     grads = []
#     for i in range(len(x)):
#         grad_i = lp_grad_fn(
#             x[i], dict(loc=loc, scale=scale, concentration=concentration)
#         )
#         grads.append(grad_i)

#     for param in ["loc", "scale", "concentration"]:
#         assert jnp.all(~jnp.isnan(lp_grad[param]))
#         assert jnp.all(~jnp.isinf(lp_grad[param]))


# nobs = 100
# key = jax.random.PRNGKey(0)  # generate PRNG keys
# k1, k2, k3, k4 = jax.random.split(key=key, num=4)

# x = tfd.Uniform(low=-2.0, high=2.0).sample(nobs, seed=k3)  # draw x


# class TestLocScaleSurvivalTransformationDist:
#     def test___init__(self):
#         nshape = 10
#         shape = sample_shape(k1, nshape=nshape, scale=0.3).sample  # set shape vector
#         log_eps = sample_simple_ptm(
#             k2, shape=(nobs,), coef=shape, centered=True, scaled=True
#         )

#         knots = OnionKnots(a=-4.0, b=4.0, nparam=nshape)
#         coef = OnionCoef(knots=knots, name="").function(shape)

#         dist = LocScaleCensoredTransformationDist(
#             knots=knots.knots,
#             coef=jnp.expand_dims(coef, 0),
#             loc=jnp.zeros(1),
#             scale=jnp.ones(1),
#         )

#         # Create a single indicator for censored data
#         censoring_indicator = tfd.Bernoulli(probs=0.15).sample(nobs, seed=k2)
#         u_lower = tfd.Uniform(low=jnp.log(0.1), high=log_eps).sample(seed=key)

#         lower_bound = jnp.full_like(log_eps, jnp.nan)
#         upper_bound = jnp.full_like(log_eps, jnp.nan)

#         lower_bound = jnp.where(censoring_indicator == 1, u_lower, lower_bound)
#         log_eps_censored = jnp.where(censoring_indicator == 1, jnp.nan, log_eps)

#         log_eps_obs = jnp.concatenate(
#             [log_eps_censored[:, None], lower_bound[:, None], upper_bound[:, None]],
#             axis=1,
#         )

#         with jax.disable_jit():
#             dist.log_prob(log_eps_obs)

#     def test_log_prob(self):
#         nshape = 10
#         shape = sample_shape(k1, nshape=nshape, scale=0.3).sample  # set shape vector
#         log_eps = sample_simple_ptm(
#             k2, shape=(nobs,), coef=shape, centered=True, scaled=True
#         )

#         knots = OnionKnots(a=-4.0, b=4.0, nparam=nshape)
#         coef = OnionCoef(knots=knots, name="").function(shape)

#         dist = CensoredTransformationDist(
#             knots=knots.knots,
#             coef=jnp.expand_dims(coef, 0),
#         )

#         # Create a single indicator for censored data
#         censoring_indicator = tfd.Bernoulli(probs=0.15).sample(nobs, seed=k2)
#         u_lower = tfd.Uniform(low=jnp.log(0.1), high=log_eps).sample(seed=key)

#         lower_bound = jnp.full_like(log_eps, jnp.nan)
#         upper_bound = jnp.full_like(log_eps, jnp.nan)

#         lower_bound = jnp.where(censoring_indicator == 1, u_lower, lower_bound)
#         log_eps_censored = jnp.where(censoring_indicator == 1, jnp.nan, log_eps)

#         log_eps_obs = jnp.concatenate(
#             [log_eps_censored[:, None], lower_bound[:, None], upper_bound[:, None]],
#             axis=1,
#         )

#         with jax.disable_jit():
#             dist.log_prob(log_eps_obs)
