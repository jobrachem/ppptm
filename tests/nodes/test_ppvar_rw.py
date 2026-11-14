import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax.random import key, normal

import ppptm as gptm

locs = gptm.unit_grid_vars()


class TestPPParam:
    @pytest.mark.skip
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


@pytest.mark.parametrize("nparam", [3, 30])
@pytest.mark.parametrize("locwise_amplitude", [False, True])
def test_random_walk_covariance(nparam, locwise_amplitude):
    locations = gptm.unit_grid_vars(ngrid=2, n_subset=2)
    amplitude = jnp.array([0.3, 0.5, 0.7, 0.9]) if locwise_amplitude else 0.7
    length_scale, salt = 0.8, 0.02
    var = gptm.RandomWalkParamPredictiveProcessGP(
        gptm.GPKernel(
            tfk.MaternFiveHalves, amplitude=amplitude, length_scale=length_scale
        ),
        locations,
        nparam,
        salt=salt,
        name="coef",
    )

    # Basis responses recover the actual node's linear map without sampling error.
    basis_values = []
    for latent in jnp.eye(2 * nparam):
        var.latent_var.value = latent
        basis_values.append(np.asarray(var.update().value))
    basis_values = np.stack(basis_values)  # (M * D, N, D)
    mapping = basis_values.reshape(2 * nparam, -1).T

    kernel = tfk.MaternFiveHalves(amplitude=1.0, length_scale=length_scale)
    inducing = locations.ordered_subset.value
    kuu = np.asarray(kernel.matrix(inducing, inducing)) + salt * np.eye(2)
    kdu = np.asarray(kernel.matrix(locations.ordered.value, inducing))
    spatial_cov = kdu @ np.linalg.solve(kuu, kdu.T)
    amplitudes = np.broadcast_to(np.asarray(amplitude), (4,))
    spatial_cov *= np.outer(amplitudes, amplitudes)
    indices = np.arange(1, nparam + 1)
    brownian_cov = np.minimum.outer(indices, indices)
    np.testing.assert_allclose(
        mapping @ mapping.T,
        np.kron(spatial_cov, brownian_cov),
        rtol=2e-5,
        atol=2e-6,
    )

    # Distinct coefficient increments are independent, with equal variance.
    increments = np.diff(basis_values, axis=-1)
    for i in range(4):
        np.testing.assert_allclose(
            increments[:, i, :].T @ increments[:, i, :],
            spatial_cov[i, i] * np.eye(nparam - 1),
            rtol=2e-5,
            atol=2e-6,
        )


def test_old_latent_conversion_preserves_coefficients():
    locations = gptm.unit_grid_vars(ngrid=2, n_subset=2)
    nparam, amplitude, salt = 5, 0.7, 0.02
    var = gptm.RandomWalkParamPredictiveProcessGP(
        gptm.GPKernel(tfk.MaternFiveHalves, amplitude=amplitude, length_scale=0.8),
        locations,
        nparam,
        salt=salt,
        name="coef",
    )
    old_latent = np.linspace(-0.3, 0.6, 2 * nparam).reshape(2, nparam)
    brownian_cov = np.minimum.outer(np.arange(1, nparam + 1), np.arange(1, nparam + 1))
    weights = np.linalg.cholesky(brownian_cov)
    spatial_factor = np.linalg.cholesky(np.asarray(var.Kuu.value) + salt * np.eye(2))
    old_coef = (
        amplitude
        * np.asarray(var.Kdu.value)
        @ np.linalg.solve(spatial_factor.T, old_latent)
        @ brownian_cov.T
    )

    var.latent_var.value = jnp.asarray((old_latent @ weights).ravel())
    np.testing.assert_allclose(var.update().value, old_coef, rtol=2e-5, atol=2e-6)


def test_random_walk_gp_applies_its_bijector():
    import tensorflow_probability.substrates.jax.bijectors as tfb

    def build(bijector):
        gp = gptm.RandomWalkParamPredictiveProcessGP(
            gptm.GPKernel(tfk.ExponentiatedQuadratic),
            gptm.unit_grid_vars(2),
            3,
            bijector=bijector,
            name="gp",
        )
        gp.latent_var.value = normal(key(4), gp.latent_var.value.shape)
        gp.update()
        return gp.value

    assert jnp.allclose(build(tfb.Exp()), jnp.exp(build(tfb.Identity())), rtol=1e-5)


def test_random_walk_sample_df_needs_two_dimensional_locations():
    lonlat = jnp.array([[-100.0, 10.0], [-90.0, 20.0], [-80.0, 30.0], [-70.0, 40.0]])
    sphere = gptm.LocationVars.new_from(lonlat, n_subset=2, from_2d_to_3d=True)
    with pytest.raises(ValueError, match="2-D"):
        gptm.GPTMCoef.sample_df(sphere, nparam=3)


def test_random_walk_samplers_default_to_the_matern_kernel():
    grid = gptm.unit_grid_vars(ngrid=3)
    default = gptm.GPTMCoef.sample_array(grid, nparam=3, n=2, seed=3)
    matern = gptm.GPTMCoef.sample_array(
        grid, nparam=3, n=2, seed=3, kernel=tfk.MaternFiveHalves
    )
    assert jnp.array_equal(default, matern)
