import jax.numpy as jnp
import pytest
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax.random import key, normal

import ppptm as gptm

locs = gptm.unit_grid_vars()


class TestPPParam:
    @pytest.mark.skip
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


def _softplus_gp(amplitude: float) -> gptm.ParamPredictiveProcessGP:
    import tensorflow_probability.substrates.jax.bijectors as tfb

    var = gptm.ParamPredictiveProcessGP(
        locs=gptm.unit_grid_vars(ngrid=4),
        gp_kernel=gptm.GPKernel(tfk.MaternFiveHalves, amplitude=jnp.array(amplitude)),
        bijector=tfb.Softplus(),
        name="scale",
    )
    var.latent_var.value = normal(key(3), var.latent_var.value.shape)
    var.mean.value = jnp.array(0.5)
    var.update()
    return var


def test_amplitude_scales_the_field_inside_the_bijector():
    # amplitude 0 means no spatial variation: the bijector of the mean everywhere
    flat = _softplus_gp(0.0)
    assert jnp.allclose(flat.value, jnp.log1p(jnp.exp(0.5)))

    # more amplitude means more variation relative to the level
    def spread(amplitude: float):
        value = _softplus_gp(amplitude).value
        return value.max() / value.min()

    assert spread(0.1) < spread(1.0) < spread(2.0)


def test_sample_df_reports_lon_lat_for_sphere_locations():
    lonlat = jnp.array([[-100.0, 10.0], [-90.0, 20.0], [-80.0, 30.0], [-70.0, 40.0]])
    sphere = gptm.LocationVars.new_from(lonlat, n_subset=2, from_2d_to_3d=True)

    df = gptm.GPVar.sample_df(sphere, n=1)
    pairs = {(round(lon, 3), round(lat, 3)) for lon, lat in zip(df["lon"], df["lat"])}
    assert pairs == {(round(float(a), 3), round(float(b), 3)) for a, b in lonlat}
