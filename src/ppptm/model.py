from __future__ import annotations

from functools import partial
from typing import Any, Literal

import jax.numpy as jnp
import liesel.model as lsl
import optax
from jax import Array
from jax.typing import ArrayLike
from liesel.goose import OptimResult, Stopper, optim_flat
from liesel_ptm.dist import (
    LocScalePseudoTransformationDist,
    LocScaleTransformationDist,
    PseudoTransformationDist,
    TransformationDist,
)

from .bspline import OnionKnots, OnionSpline
from .marginals import G, H
from .nodes.ppvar_rw import SpatPTMCoef
from .util.locs import LocationVars

KeyArray = Any


class HDist(lsl.Dist):
    """
    Distribution wrapper that builds a transformation distribution
    using a spline-based transformation.

    Parameters
    ----------
    knots
        Spline knot sequence.
    coef
        Coefficients for the transformation.
    centered
        If True, use centered parameterization.
    scaled
        If True, use scaled parameterization.
    bspline
        Which spline variant to use: 'onion' or 'identity'.
    **G_kwargs
        Parameters of the parametric distribution.
    **lsl_dist_kwargs
        Forwarded to the parent distribution constructor.

    Attributes
    ----------
    partial_dist_class
        Partial distribution class used to construct per-observation distributions.
    """

    def __init__(
        self,
        knots: Array,
        coef: SpatPTMCoef | lsl.Var,
        g_dist: lsl.Dist,
        centered: bool = False,
        scaled: bool = False,
        bspline: Literal["onion", "identity"] = "onion",
        locscale: bool = False,
        _name: str = "",
    ) -> None:
        if len(g_dist.inputs) > 0:
            raise ValueError(
                f"Positional inputs found on {g_dist=}. {self} requires distribution to "
                "be defined with keyword inputs."
            )

        trafo_dist = LocScaleTransformationDist if locscale else TransformationDist

        match bspline:
            case "onion":
                bspline_inst = OnionSpline(knots)
                partial_dist_class = partial(
                    trafo_dist,
                    parametric_distribution=g_dist.distribution,
                    bspline=bspline_inst,
                    centered=centered,
                    scaled=scaled,
                    batched=True,
                )
            case "identity":
                dist_class = (
                    LocScalePseudoTransformationDist
                    if locscale
                    else PseudoTransformationDist
                )
                bspline_inst = OnionSpline(knots)
                dist_class.bspline = bspline_inst
                partial_dist_class = partial(
                    dist_class,
                    parametric_distribution=g_dist.distribution,
                    centered=centered,
                    scaled=scaled,
                    batched=True,
                )

        self.partial_dist_class = partial_dist_class
        self.bspline = bspline_inst

        super().__init__(
            partial_dist_class,
            _name=_name,
            _needs_seed=False,
            coef=coef,
            **dict(g_dist.kwinputs),
        )


class Model:
    def __init__(
        self,
        y: ArrayLike,
        locs: LocationVars,
        knots: ArrayLike,
        coef: SpatPTMCoef | lsl.Var | None = None,
        g_dist: lsl.Dist | None = None,
        to_float32: bool = False,
        bspline: Literal["onion", "identity"] = "onion",
        locscale: bool = False,
    ):
        knots = jnp.asarray(knots)
        if g_dist is None:
            g_dist = G(y, locs).new_gaussian()
            locscale = True

        if coef is None:
            coef = H(locs, nparam=knots.size - 11).new_coef()

        nloc1, D = coef.value.shape
        nobs, nloc2 = jnp.shape(y)

        if not nloc1 == nloc2:
            raise ValueError(
                "Different numbers of locations found for "
                f"response ({nloc2}) and coef ({nloc1})."
            )

        self.locscale = locscale
        self.D = D
        self.nobs = nobs
        self.nloc = nloc1
        self.knots = knots
        self.coef = coef
        self.g_dist = g_dist

        dist = HDist(
            knots=self.knots,
            bspline=bspline,
            g_dist=g_dist,
            coef=coef,
            centered=False,
            scaled=False,
            locscale=locscale,
        )

        self.dist_node = dist

        self.response = lsl.Var.new_obs(
            value=jnp.asarray(y), distribution=dist, name="response"
        ).update()

        self.response.update()

        self._to_float32 = to_float32

        self.graph = lsl.Model([self.response], to_float32=self._to_float32)

    @classmethod
    def new_HG(
        cls,
        y: ArrayLike,
        locs: LocationVars,
        a: float = -7.0,
        b: float = 7.0,
        nparam: int = 40,
        g_dist: lsl.Dist | None = None,
        coef: SpatPTMCoef | None = None,
        locscale: bool = False,
    ) -> Model:
        knots = OnionKnots(a=a, b=b, nparam=nparam)

        model = cls(
            y=y,
            locs=locs,
            knots=knots.knots,
            coef=coef,
            bspline="onion",
            locscale=locscale,
            g_dist=g_dist,
        )
        return model

    @classmethod
    def new_G(
        cls,
        y: ArrayLike,
        locs: LocationVars,
        g_dist: lsl.Dist | None = None,
        locscale: bool = False,
    ) -> Model:
        knots = OnionKnots(a=-1.0, b=1.0, nparam=3)
        coef = lsl.Var.new_value(jnp.zeros((jnp.shape(y)[-1], knots.nparam)))

        model = cls(
            y=y,
            locs=locs,
            knots=knots.knots,
            coef=coef,
            bspline="identity",
            locscale=locscale,
            g_dist=g_dist,
        )
        return model

    @property
    def parameters(self):
        return list(self.graph.parameters)

    def fit(
        self,
        stopper: Stopper | None = None,
        response_validation: ArrayLike | None = None,
        optimizer: optax.GradientTransformation | None = None,
        progress_bar: bool = False,
        **kwargs,
    ) -> OptimResult:
        if response_validation is not None:
            _, varval = self.graph.copy_nodes_and_vars()
            varval["response"].value = jnp.asarray(response_validation)
            model_validation = lsl.Model(
                [varval["response"]], to_float32=self._to_float32
            )
        else:
            model_validation = self.graph

        result = optim_flat(
            model_train=self.graph,
            params=self.parameters,
            model_validation=model_validation,
            stopper=stopper,
            progress_bar=progress_bar,
            optimizer=optimizer,
            **kwargs,
        )

        self.fit_result = result
        self.graph.state = self.graph.update_state(self.fit_result.position)

        return result

    def init_dist(
        self,
        samples: dict[str, Array] | None = None,
    ) -> TransformationDist | LocScaleTransformationDist | PseudoTransformationDist:
        if samples is None:
            assert self.response.dist_node is not None
            return self.response.dist_node.init_dist()

        assert samples is not None
        pred = self.graph.predict(samples=samples)
        coef = pred.pop(self.coef.name)

        kwargs_G = {}
        for param_name in self.g_dist.kwinputs:
            kwargs_G[param_name] = pred.pop(param_name)

        return self.dist_node.partial_dist_class(coef=coef, **kwargs_G)

    def g(self, y: ArrayLike, samples: dict[str, Array] | None = None) -> Array:
        dist = self.init_dist(samples)
        return dist.transformation_and_logdet_parametric(y)[0]

    def gi(self, yt: ArrayLike, samples: dict[str, Array] | None = None) -> Array:
        dist = self.init_dist(samples)
        return dist.inverse_transformation_parametric(yt)

    def h(self, yt: ArrayLike, samples: dict[str, Array] | None = None) -> Array:
        dist = self.init_dist(samples)
        zt = dist.transformation_and_logdet_spline(yt)[0]
        return zt

    def hi(self, zt: ArrayLike, samples: dict[str, Array] | None = None) -> Array:
        dist = self.init_dist(samples)
        yt = dist.inverse_transformation_spline(zt)
        return yt

    def hg(self, y: ArrayLike, samples: dict[str, Array] | None = None) -> Array:
        dist = self.init_dist(samples)
        zt = dist.transformation_and_logdet(y)[0]
        return zt

    def hgi(self, zt: ArrayLike, samples: dict[str, Array] | None = None) -> Array:
        dist = self.init_dist(samples)
        y = dist.inverse_transformation(zt)
        return y

    def log_prob(self, y: ArrayLike, samples: dict[str, Array] | None = None) -> Array:
        dist = self.init_dist(samples)
        lp = dist.log_prob(y)
        return lp
