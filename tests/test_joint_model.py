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
