import jax.numpy as jnp
import liesel.model as lsl
import liesel_ptm as ptm
import pytest
import tensorflow_probability.substrates.jax.distributions as tfd
from jax.random import key
from liesel_ptm.dist import PseudoTransformationDist

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
        coef: lsl.Var | None = (
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


OBSERVED_CASES = tuple(
    pytest.param(
        configuration,
        value_name,
        value,
        expected_shape,
        surface,
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


SCALAR_CASES = tuple(
    pytest.param(
        configuration,
        surface,
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


def test_replaced_distribution_input_matches_graph_likelihood():
    model = gptm.Model.new_G(RESPONSE, gptm.unit_grid_vars(ngrid=2))
    response_dist = model.response.dist_node
    assert response_dist is not None

    covariate = lsl.Var.new_obs(jnp.zeros(4), name="covariate")
    effect = lsl.Var.new_param(jnp.array(0.0), name="covariate_effect")
    loc_predictor = lsl.Var(
        lsl.Calc(
            lambda loc, x, beta: loc + x * beta,
            response_dist["loc"],
            covariate,
            effect,
        ),
        name="loc_predictor",
    )
    response_dist["loc"] = loc_predictor

    samples = {
        name: jnp.broadcast_to(var.value, (2,) + jnp.shape(var.value))
        for name, var in model.graph.parameters.items()
    }
    samples[effect.name] = jnp.array([0.0, 1.0])
    newdata = {covariate.name: jnp.ones(4)}

    actual = model.log_prob(RESPONSE, samples=samples, newdata=newdata).sum(
        axis=(-2, -1)
    )
    expected = jnp.stack(
        [
            model.graph.update_state(
                {name: value[i] for name, value in samples.items()} | newdata
            )["_model_log_lik"].value
            for i in range(2)
        ]
    )

    assert jnp.allclose(actual, expected)


def test_unbatched_position_matches_graph_likelihood_exactly():
    response = jnp.linspace(0.1, 1.2, 48).reshape(3, 16)
    model = gptm.Model.new_G(response, gptm.unit_grid_vars(ngrid=4))
    response_dist = model.response.dist_node
    assert response_dist is not None

    covariate = lsl.Var.new_obs(jnp.zeros(16), name="covariate")
    effect = lsl.Var.new_param(jnp.array(0.0), name="covariate_effect")
    loc_predictor = lsl.Var(
        lsl.Calc(
            lambda loc, x, beta: loc + x * beta,
            response_dist["loc"],
            covariate,
            effect,
        ),
        name="loc_predictor",
    )
    response_dist["loc"] = loc_predictor

    position = {name: var.value for name, var in model.graph.parameters.items()}
    for name, value in position.items():
        if "latent" in name:
            position[name] = jnp.linspace(-0.2, 0.2, value.size).reshape(value.shape)
    position[effect.name] = jnp.array(1.0)
    newdata = {covariate.name: jnp.ones(16)}

    state = model.graph.update_state(position | newdata)
    actual = model.log_prob(response, samples=position, newdata=newdata).sum()

    assert jnp.array_equal(actual, state["_model_log_lik"].value)


NEW_LOCATION_CASES = tuple(
    pytest.param(
        configuration,
        surface,
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
        assert jnp.allclose(result, expected, atol=2e-5, rtol=0.0)


@pytest.mark.parametrize("constructor", ["direct", "new_HG", "new_G"])
@pytest.mark.parametrize(
    "centered,scaled", [(False, False), (True, False), (False, True), (True, True)]
)
def test_spline_normalization_flags(constructor, centered, scaled):
    locations = gptm.unit_grid_vars(ngrid=2)
    params = {
        name: jnp.full(4, value)
        for name, value in {
            "loc": 0.2,
            "scale": 1.1,
            "skewness": 1.3,
            "df": 8.0,
        }.items()
    }
    g_dist = lsl.Dist(
        tfd.TwoPieceStudentT,
        loc=params["loc"],
        scale=params["scale"],
        skewness=params["skewness"],
        df=params["df"],
    )
    response = RESPONSE.at[0, 0].set(jnp.nan)
    knots = ptm.OnionKnots(a=-7.0, b=7.0, nparam=4)
    if constructor == "new_G":
        model = gptm.Model.new_G(
            response,
            locations,
            g_dist=g_dist,
            centered=centered,
            scaled=scaled,
            mask_nan_response=True,
        )
        expected = PseudoTransformationDist(
            coef=jnp.array(0.0),
            parametric_distribution=tfd.TwoPieceStudentT,
            centered=centered,
            scaled=scaled,
            loc=params["loc"],
            scale=params["scale"],
            skewness=params["skewness"],
            df=params["df"],
        )
    else:
        coef = lsl.Var.new_value(ONION_COEF, name="coef")
        if constructor == "direct":
            model = gptm.Model(
                response,
                locations,
                knots=knots.knots,
                coef=coef,
                g_dist=g_dist,
                centered=centered,
                scaled=scaled,
                mask_nan_response=True,
            )
        else:
            model = gptm.Model.new_HG(
                response,
                locations,
                nparam=4,
                coef=coef,
                g_dist=g_dist,
                centered=centered,
                scaled=scaled,
                mask_nan_response=True,
            )
        expected = ptm.onion_dist(
            knots=knots.knots,
            loc_scale=False,
            parametric_distribution=tfd.TwoPieceStudentT,
            centered=centered,
            scaled=scaled,
        )(coef=ONION_COEF, **params)

    dist = model.init_dist()
    assert dist.centered is centered
    assert dist.scaled is scaled
    log_prob = model.log_prob(RESPONSE)
    assert log_prob.shape == RESPONSE.shape
    assert jnp.allclose(log_prob, expected.log_prob(RESPONSE), atol=2e-5)
    transformed = model.hg(RESPONSE)
    assert jnp.allclose(
        transformed, expected.transformation_and_logdet(RESPONSE)[0], atol=2e-5
    )
    assert jnp.allclose(model.hgi(transformed), RESPONSE, atol=2e-5)
    masked_log_prob = expected.log_prob(RESPONSE).at[0, 0].set(0.0)
    assert jnp.allclose(model.response.log_prob, masked_log_prob, atol=2e-5)
    assert jnp.allclose(model.graph.log_lik, masked_log_prob.sum(), atol=2e-5)

    if constructor == "new_HG":
        samples = {"coef": jnp.stack([ONION_COEF, 1.5 * ONION_COEF])}
        predicted_dist = model.init_dist(samples=samples)
        assert predicted_dist.centered is centered
        assert predicted_dist.scaled is scaled
        predicted_log_prob = model.log_prob(RESPONSE, samples=samples)
        assert predicted_log_prob.shape == (2,) + RESPONSE.shape
        for i, coefficients in enumerate(samples["coef"]):
            reference = ptm.onion_dist(
                knots=knots.knots,
                loc_scale=False,
                parametric_distribution=tfd.TwoPieceStudentT,
                centered=centered,
                scaled=scaled,
            )(coef=coefficients, **params)
            assert jnp.allclose(
                predicted_log_prob[i], reference.log_prob(RESPONSE), atol=2e-5
            )
