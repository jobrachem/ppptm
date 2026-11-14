import jax.numpy as jnp
import pytest
from jax.random import key, normal

import ppptm as gptm


def test_dependence_model_rejects_missing_responses():
    response = normal(key(1), (6, 9)).at[0, 0].set(jnp.nan)
    marginal = gptm.Model.new_G(
        response, gptm.unit_grid_vars(ngrid=3), mask_nan_response=True
    )
    composite = gptm.CompositeTransformations(marginal)

    with pytest.raises(ValueError, match="missing"):
        composite.init_dependence_model(largest_conditioning_set=3)


def test_sampling_without_marginal_keeps_the_raw_scale():
    response = 10.0 + 3.0 * normal(key(2), (6, 9))
    marginal = gptm.Model.new_G(response, gptm.unit_grid_vars(ngrid=3))
    composite = gptm.CompositeTransformations(marginal, use_marginal=False)
    composite.init_dependence_model(largest_conditioning_set=3)

    fixed = response[0, :3]
    draws = composite.sample(key(0), n=2, fixed_y=fixed)
    assert jnp.allclose(draws[:, :3], fixed, atol=1e-5)
