import jax
import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd

from ppptm.identity import (
    IdentityTransformationDist,
    LocScaleIdentityTransformationDist,
)

LOC = jnp.array([0.0, 0.5, -1.0])
SCALE = jnp.array([1.0, 2.0, 0.5])
VALUE = jnp.array([[0.3, -0.2, -0.6], [-1.1, 2.0, -1.2]])


def _dists():
    return [
        IdentityTransformationDist(
            coef=jnp.array(0.0),
            parametric_distribution=tfd.Normal,
            loc=LOC,
            scale=SCALE,
        ),
        LocScaleIdentityTransformationDist(coef=jnp.array(0.0), loc=LOC, scale=SCALE),
    ]


@pytest.mark.parametrize("dist", _dists(), ids=["generic", "location-scale"])
def test_identity_dist_matches_parametric_distribution(dist):
    parametric = tfd.Normal(LOC, SCALE)

    assert dist.batch_shape == (3,)
    assert jnp.allclose(dist.log_prob(VALUE), parametric.log_prob(VALUE), atol=1e-5)
    assert jnp.allclose(dist.mean(), LOC)
    assert jnp.allclose(dist.variance(), SCALE**2)

    h, logdet = dist.transformation_and_logdet_spline(VALUE)
    assert jnp.array_equal(h, VALUE)
    assert jnp.array_equal(logdet, jnp.zeros_like(VALUE))
    assert jnp.array_equal(dist.inverse_transformation_spline(VALUE), VALUE)
    z, _ = dist.transformation_and_logdet(VALUE)
    assert jnp.allclose(dist.inverse_transformation(z), VALUE, atol=1e-5)
    assert dist.moment_quadrature_diagnostic()["ok"]


@pytest.mark.parametrize("dist", _dists(), ids=["generic", "location-scale"])
def test_identity_dist_survives_rebuilds(dist):
    expected = dist.log_prob(VALUE)

    copied = dist.copy()
    assert type(copied) is type(dist)
    assert jnp.array_equal(copied.log_prob(VALUE), expected)

    sliced = dist[1:]
    assert type(sliced) is type(dist)
    assert jnp.allclose(sliced.log_prob(VALUE[:, 1:]), expected[:, 1:])

    jitted = jax.jit(lambda d, x: d.log_prob(x))(dist, VALUE)
    assert jnp.allclose(jitted, expected)
