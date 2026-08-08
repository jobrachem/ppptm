from typing import Any

import jax.numpy as jnp
import liesel.model as lsl
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.random import key

import ppptm as gptm

CONFIGURATIONS = (
    ("onion", False),
    ("onion", True),
    ("identity", False),
    ("identity", True),
)
SURFACES = ("g", "gi", "h", "hi", "hg", "hgi", "log_prob")

RESPONSE = jnp.array(
    [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ],
    dtype=jnp.float32,
)
LOCATION_VALUES = jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
NEW_LOCATIONS = jnp.array(
    [[0.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [1.0, 1.0]],
    dtype=jnp.float32,
)
NEW_RESPONSE = jnp.array(
    [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.9, 1.0, 1.1, 1.2, 1.3],
    ],
    dtype=jnp.float32,
)
GOLDEN_VALUES = jnp.array(
    [[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5]], dtype=jnp.float32
)
ONION_GOLDEN_VALUES = RESPONSE[:2]
ONION_COEF = jnp.array(
    [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.1, 0.0, -0.1],
        [0.0, 0.1, 0.2, 0.3],
        [-0.1, 0.0, 0.1, 0.2],
    ],
    dtype=jnp.float32,
)

TICKET_03_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="Ticket 03 must align public posterior axes and expose newdata.",
)


def _configuration_id(configuration: tuple[str, bool]) -> str:
    spline, locscale = configuration
    return f"{spline}-{'location-scale' if locscale else 'generic'}"


def _normal_dist() -> lsl.Dist:
    return lsl.Dist(tfd.Normal, loc=jnp.array(0.0), scale=jnp.array(1.0))


def _new_model(
    configuration: tuple[str, bool],
    *,
    fixed: bool = False,
    mask_nan_response=False,
) -> gptm.Model:
    spline, locscale = configuration
    locs = gptm.unit_grid_vars(ngrid=2)

    if fixed:
        g_dist = _normal_dist()
        coef: Any = (
            lsl.Var.new_value(ONION_COEF, name="coef") if spline == "onion" else None
        )
    else:
        g_dist = gptm.G(RESPONSE, locs).new_gaussian()
        coef = None

    if spline == "identity":
        return gptm.Model.new_G(
            RESPONSE,
            locs,
            g_dist=g_dist,
            locscale=locscale,
            mask_nan_response=mask_nan_response,
        )

    return gptm.Model.new_HG(
        RESPONSE,
        locs,
        nparam=4,
        g_dist=g_dist,
        coef=coef,
        locscale=locscale,
        mask_nan_response=mask_nan_response,
    )


def _call_public_surface(
    model: gptm.Model,
    surface: str,
    value,
    *,
    samples=None,
    newdata=None,
):
    kwargs = {}
    if samples is not None:
        kwargs["samples"] = samples
    if newdata is not None:
        kwargs["newdata"] = newdata
    return getattr(model, surface)(value, **kwargs)


@pytest.fixture(scope="module")
def model_for():
    models = {}

    def get(configuration, *, fixed=False, mask_nan_response=False):
        cache_key = (configuration, fixed, mask_nan_response)
        if cache_key not in models:
            models[cache_key] = _new_model(
                configuration,
                fixed=fixed,
                mask_nan_response=mask_nan_response,
            )
        return models[cache_key]

    return get


def _observed_case_is_gap(configuration, value_name, surface):
    spline, _ = configuration
    return (
        spline == "onion"
        and value_name == "location-vector"
        and surface in {"h", "hi", "hg", "hgi", "log_prob"}
    )


OBSERVED_CASES = tuple(
    pytest.param(
        configuration,
        value_name,
        value,
        expected_shape,
        surface,
        marks=(
            TICKET_03_XFAIL
            if _observed_case_is_gap(configuration, value_name, surface)
            else ()
        ),
        id=f"{_configuration_id(configuration)}-{value_name}-{surface}",
    )
    for configuration in CONFIGURATIONS
    for value_name, value, expected_shape in (
        ("location-vector", LOCATION_VALUES, (4,)),
        ("observation-by-location", RESPONSE, (3, 4)),
    )
    for surface in SURFACES
)


@pytest.mark.parametrize(
    ("configuration", "value_name", "value", "expected_shape", "surface"),
    OBSERVED_CASES,
)
def test_observed_public_shapes(
    model_for, configuration, value_name, value, expected_shape, surface
):
    result = _call_public_surface(model_for(configuration), surface, value)
    assert jnp.shape(result) == expected_shape
    assert jnp.all(jnp.isfinite(result))


def _scalar_case_is_gap(configuration, surface):
    spline, _ = configuration
    if spline == "identity":
        return surface in {"h", "hi"}
    return surface in {"h", "hi", "hg", "hgi", "log_prob"}


SCALAR_CASES = tuple(
    pytest.param(
        configuration,
        surface,
        marks=(TICKET_03_XFAIL if _scalar_case_is_gap(configuration, surface) else ()),
        id=f"{_configuration_id(configuration)}-{surface}",
    )
    for configuration in CONFIGURATIONS
    for surface in SURFACES
)


@pytest.mark.parametrize(("configuration", "surface"), SCALAR_CASES)
def test_scalar_public_shapes(model_for, configuration, surface):
    result = _call_public_surface(model_for(configuration), surface, jnp.array(0.25))
    assert jnp.shape(result) == (4,)
    assert jnp.all(jnp.isfinite(result))


POSTERIOR_CASES = tuple(
    pytest.param(
        configuration,
        sample_shape,
        surface,
        marks=TICKET_03_XFAIL,
        id=f"{_configuration_id(configuration)}-{sample_shape}-{surface}",
    )
    for configuration in CONFIGURATIONS
    for sample_shape in ((2,), (2, 3))
    for surface in SURFACES
)


@pytest.mark.parametrize(
    ("configuration", "sample_shape", "surface"),
    POSTERIOR_CASES,
)
def test_posterior_public_shapes(model_for, configuration, sample_shape, surface):
    model = model_for(configuration)
    samples = model.graph.sample(sample_shape, seed=key(11))
    result = _call_public_surface(model, surface, RESPONSE, samples=samples)
    assert jnp.shape(result) == sample_shape + (3, 4)
    assert jnp.all(jnp.isfinite(result))


@TICKET_03_XFAIL
def test_identity_posterior_values_and_axis_order():
    locs = gptm.unit_grid_vars(ngrid=2)
    g_dist = lsl.Dist(
        tfd.Normal,
        loc=lsl.Var.new_value(jnp.zeros(4), name="loc"),
        scale=lsl.Var.new_value(jnp.ones(4), name="scale"),
    )
    model = gptm.Model.new_G(RESPONSE, locs, g_dist=g_dist, locscale=True)
    samples = model.graph.sample((2, 3), seed=key(13))
    expected = jnp.broadcast_to(RESPONSE, (2, 3, 3, 4))

    for surface in ("g", "gi", "h", "hi", "hg", "hgi"):
        result = _call_public_surface(model, surface, RESPONSE, samples=samples)
        assert jnp.array_equal(result, expected)

    expected_log_prob = jnp.broadcast_to(
        tfd.Normal(loc=0.0, scale=1.0).log_prob(RESPONSE), expected.shape
    )
    result = _call_public_surface(model, "log_prob", RESPONSE, samples=samples)
    assert jnp.array_equal(result, expected_log_prob)


NEW_LOCATION_CASES = tuple(
    pytest.param(
        configuration,
        surface,
        marks=TICKET_03_XFAIL,
        id=f"{_configuration_id(configuration)}-{surface}",
    )
    for configuration in CONFIGURATIONS
    for surface in SURFACES
)


@pytest.mark.parametrize(("configuration", "surface"), NEW_LOCATION_CASES)
def test_predicted_location_public_shapes(model_for, configuration, surface):
    model = model_for(configuration)
    samples = model.graph.sample((2,), seed=key(12))
    result = _call_public_surface(
        model,
        surface,
        NEW_RESPONSE,
        samples=samples,
        newdata={"sample_locs": NEW_LOCATIONS},
    )
    assert jnp.shape(result) == (2, 3, 5)
    assert jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize(
    "configuration",
    CONFIGURATIONS,
    ids=_configuration_id,
)
def test_observed_inverse_round_trips(model_for, configuration):
    model = model_for(configuration, fixed=True)

    for forward, inverse in (("g", "gi"), ("h", "hi"), ("hg", "hgi")):
        transformed = _call_public_surface(model, forward, RESPONSE)
        restored = _call_public_surface(model, inverse, transformed)
        assert jnp.allclose(restored, RESPONSE, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize(
    "configuration",
    CONFIGURATIONS,
    ids=_configuration_id,
)
def test_finite_likelihood_and_nan_masking(model_for, configuration):
    response = RESPONSE.at[0, 0].set(jnp.nan).at[1, 3].set(jnp.nan)
    model = model_for(configuration, fixed=True, mask_nan_response=True)

    log_prob = model.log_prob(response)
    mask = jnp.isnan(response)

    assert jnp.all(jnp.isfinite(log_prob))
    assert jnp.all(log_prob[mask] == 0.0)
    assert jnp.all(jnp.isfinite(log_prob[~mask]))


IDENTITY_GOLDENS = {
    False: {
        "g": jnp.array(
            [
                [-0.9999997, -0.49999997, 0.0, 0.49999997],
                [0.9999997, 0.49999997, 0.0, -0.49999997],
            ],
            dtype=jnp.float32,
        ),
        "gi": jnp.array(
            [
                [-0.9999997, -0.49999997, 0.0, 0.49999997],
                [0.9999997, 0.49999997, 0.0, -0.49999997],
            ],
            dtype=jnp.float32,
        ),
        "h": GOLDEN_VALUES,
        "hi": GOLDEN_VALUES,
        "hg": jnp.array(
            [
                [-0.9999998, -0.49999997, 0.0, 0.49999997],
                [0.9999998, 0.49999997, 0.0, -0.49999997],
            ],
            dtype=jnp.float32,
        ),
        "hgi": jnp.array(
            [
                [-0.9999998, -0.49999997, 0.0, 0.49999997],
                [0.9999998, 0.49999997, 0.0, -0.49999997],
            ],
            dtype=jnp.float32,
        ),
        "log_prob": jnp.array(
            [
                [-1.4189385, -1.0439385, -0.9189385, -1.0439385],
                [-1.4189385, -1.0439385, -0.9189385, -1.0439385],
            ],
            dtype=jnp.float32,
        ),
    },
    True: {
        "g": jnp.array(
            [[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5]],
            dtype=jnp.float32,
        ),
        "gi": jnp.array(
            [[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5]],
            dtype=jnp.float32,
        ),
        "h": jnp.array(
            [[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5]],
            dtype=jnp.float32,
        ),
        "hi": jnp.array(
            [[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5]],
            dtype=jnp.float32,
        ),
        "hg": jnp.array(
            [[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5]],
            dtype=jnp.float32,
        ),
        "hgi": jnp.array(
            [[-1.0, -0.5, 0.0, 0.5], [1.0, 0.5, 0.0, -0.5]],
            dtype=jnp.float32,
        ),
        "log_prob": jnp.array(
            [
                [-1.4189385, -1.0439385, -0.9189385, -1.0439385],
                [-1.4189385, -1.0439385, -0.9189385, -1.0439385],
            ],
            dtype=jnp.float32,
        ),
    },
}


@pytest.mark.parametrize(
    "locscale",
    (False, True),
    ids=("generic", "location-scale"),
)
def test_identity_deterministic_goldens(model_for, locscale):
    model = model_for(("identity", locscale), fixed=True)

    for surface, expected in IDENTITY_GOLDENS[locscale].items():
        assert jnp.array_equal(
            _call_public_surface(model, surface, GOLDEN_VALUES), expected
        )


ONION_GOLDENS = {
    "g": jnp.array(
        [
            [0.09999997, 0.19999996, 0.29999995, 0.39999998],
            [0.49999997, 0.59999996, 0.70000005, 0.7999999],
        ],
        dtype=jnp.float32,
    ),
    "gi": jnp.array(
        [
            [0.09999997, 0.19999996, 0.29999995, 0.39999998],
            [0.49999997, 0.59999996, 0.70000005, 0.7999999],
        ],
        dtype=jnp.float32,
    ),
    "h": jnp.array(
        [
            [-0.32672358, 0.6246085, -0.12601042, -0.02500057],
            [0.07642698, 1.0158672, 0.28062677, 0.38336658],
        ],
        dtype=jnp.float32,
    ),
    "hi": jnp.array(
        [
            [0.5231735, -0.22648959, 0.71889454, 0.8161494],
            [0.91299284, 0.17506702, 1.1054785, 1.2011368],
        ],
        dtype=jnp.float32,
    ),
    "hg": jnp.array(
        [
            [-0.32672358, 0.624609, -0.12601042, -0.02500057],
            [0.07642651, 1.0158672, 0.28062677, 0.38336706],
        ],
        dtype=jnp.float32,
    ),
    "hgi": jnp.array(
        [
            [0.5231736, -0.22648954, 0.71889454, 0.8161495],
            [0.9129927, 0.17506666, 1.1054783, 1.2011366],
        ],
        dtype=jnp.float32,
    ),
    "log_prob": jnp.array(
        [
            [-0.9730211, -1.1275647, -0.9190186, -0.9071066],
            [-0.90542966, -1.46563, -0.9333117, -0.9631347],
        ],
        dtype=jnp.float32,
    ),
}


@pytest.mark.parametrize(
    "locscale",
    (False, True),
    ids=("generic", "location-scale"),
)
def test_onion_migration_goldens(model_for, locscale):
    model = model_for(("onion", locscale), fixed=True)

    for surface, expected in ONION_GOLDENS.items():
        result = _call_public_surface(model, surface, ONION_GOLDEN_VALUES)
        assert jnp.allclose(result, expected, atol=1e-5, rtol=0.0)
