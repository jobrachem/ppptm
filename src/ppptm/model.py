from __future__ import annotations

from functools import partial
from typing import Any, Literal, cast

import jax.numpy as jnp
import liesel.model as lsl
import optax
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import Array
from jax.typing import ArrayLike
from liesel import optim as loptim
from liesel.optim.state import OptimResult
from liesel.optim.types import Position
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
type PTMDist = (
    TransformationDist
    | LocScaleTransformationDist
    | PseudoTransformationDist
    | LocScalePseudoTransformationDist
)
type TrainMonitor = Literal[
    "auto", "epoch_average", "weighted_epoch_average", "full_data"
]


def _coerce_stopper(stopper: Any | None) -> loptim.Stopper:
    if stopper is None:
        return loptim.Stopper(epochs=1000, patience=10, rtol=1e-6)

    if isinstance(stopper, loptim.Stopper):
        return stopper

    epochs = getattr(stopper, "epochs", None)
    if epochs is None:
        epochs = getattr(stopper, "max_iter", None)

    patience = getattr(stopper, "patience", None)
    if epochs is None or patience is None:
        raise TypeError(
            "stopper must be a liesel.optim.Stopper or expose max_iter/epochs "
            "and patience attributes."
        )

    return loptim.Stopper(
        epochs=int(epochs),
        patience=int(patience),
        atol=float(getattr(stopper, "atol", 0.0)),
        rtol=float(getattr(stopper, "rtol", 0.0)),
    )


def _validate_location_batch_size(
    batch_size: int | None, nloc: int, shuffle_batches: bool
) -> None:
    if batch_size is None:
        return

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be a positive integer or None.")

    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer or None.")

    if batch_size > nloc:
        raise ValueError(
            "Location batch_size must not exceed the number of locations; "
            f"got batch_size={batch_size} and nloc={nloc}."
        )

    if nloc % batch_size != 0 and not shuffle_batches:
        raise ValueError(
            "Location batch_size must divide the number of locations exactly when "
            "shuffle_batches=False, because Liesel drops incomplete remainder "
            f"batches; got batch_size={batch_size} and nloc={nloc}."
        )


def _validate_progress_n_updates(progress_n_updates: int) -> None:
    if isinstance(progress_n_updates, bool) or progress_n_updates < 1:
        raise ValueError("progress_n_updates must be a positive integer.")


def _safe_mask_value(value: ArrayLike) -> Array:
    value = jnp.asarray(value)
    finite = jnp.isfinite(value)
    finite_count = jnp.sum(finite)
    finite_sum = jnp.sum(jnp.where(finite, value, jnp.zeros_like(value)))
    fallback = finite_sum / jnp.maximum(finite_count, 1)
    return jnp.where(finite, value, fallback)


def mask_distribution(dist: tfd.Distribution, value: ArrayLike) -> tfd.Distribution:
    value = jnp.asarray(value)
    return tfd.Masked(
        distribution=dist,
        validity_mask=~jnp.isnan(value),
        safe_sample_fn=lambda _: _safe_mask_value(value),
    )


def _validate_g_dist_values_are_finite(g_dist: lsl.Dist) -> None:
    for name, node in g_dist.kwinputs.items():
        value = jnp.asarray(node.value)
        if bool(jnp.any(~jnp.isfinite(value))):
            raise ValueError(
                "Cannot enable mask_nan_response with non-finite values in "
                f"g_dist keyword input {name!r}. Provide a finite g_dist or let "
                "Model initialize the default g_dist from the response."
            )


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
        mask_nan_response: bool = False,
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
        self.mask_nan_response = mask_nan_response

        super().__init__(
            partial_dist_class,
            _name=_name,
            _needs_seed=False,
            coef=coef,
            **cast(dict[str, Any], dict(g_dist.kwinputs)),
        )

    def init_base_dist(self) -> PTMDist:
        args = [_input.value for _input in self.inputs]
        kwargs = {kw: _input.value for kw, _input in self.kwinputs.items()}
        return self.distribution(*args, **kwargs)

    def init_dist(self) -> tfd.Distribution:
        dist = self.init_base_dist()
        if not self.mask_nan_response:
            return dist

        if self.at is None:
            raise RuntimeError(f"{self!r} cannot derive a NaN mask without `at`.")

        return mask_distribution(dist, self.at.value)


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
        mask_nan_response: bool = False,
        auto_update: bool = True,
    ):
        knots = jnp.asarray(knots)
        if g_dist is None:
            g_dist = G(y, locs).new_gaussian()
            locscale = True
        elif mask_nan_response:
            _validate_g_dist_values_are_finite(g_dist)

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
        self.locs = locs
        self.mask_nan_response = mask_nan_response

        dist = HDist(
            knots=self.knots,
            bspline=bspline,
            g_dist=g_dist,
            coef=coef,
            centered=False,
            scaled=False,
            locscale=locscale,
            mask_nan_response=mask_nan_response,
        )

        self.dist_node = dist

        self.response = lsl.Var.new_obs(
            value=jnp.asarray(y), distribution=dist, name="response"
        ).update()

        self.response.update()

        self._to_float32 = to_float32

        self.graph = lsl.Model(lsl.Var(0.0, name="stub"), to_float32=to_float32)
        self.graph.auto_update = auto_update
        self.graph.replace("stub", self.response)

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
        mask_nan_response: bool = False,
        auto_update: bool = True,
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
            mask_nan_response=mask_nan_response,
            auto_update=auto_update,
        )
        return model

    @classmethod
    def new_G(
        cls,
        y: ArrayLike,
        locs: LocationVars,
        g_dist: lsl.Dist | None = None,
        locscale: bool = False,
        mask_nan_response: bool = False,
        auto_update: bool = True,
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
            mask_nan_response=mask_nan_response,
            auto_update=auto_update,
        )
        return model

    @property
    def parameters(self):
        return list(self.graph.parameters)

    @property
    def spatial_coef(self) -> SpatPTMCoef:
        if not isinstance(self.coef, SpatPTMCoef):
            raise TypeError("This model does not have spatial coefficients.")
        return self.coef

    def fit(
        self,
        stopper: Any | None = None,
        response_validation: ArrayLike | None = None,
        optimizer: optax.GradientTransformation | None = None,
        progress_bar: bool = False,
        batch_size: int | None = None,
        seed: int | None = None,
        shuffle_batches: bool = True,
        scale_loss: bool = False,
        validation_strategy: Literal["log_lik", "log_prob"] = "log_lik",
        train_monitor: TrainMonitor = "auto",
        save_position_history: bool = True,
        progress_n_updates: int = 100,
    ) -> OptimResult:
        """
        Fit the model parameters with :class:`liesel.optim.LieselOptim`.

        The training/validation split is defined over response rows, i.e. axis 0
        of arrays with shape ``(nobs, nloc)``. Mini-batching, if requested, is
        location-based instead: ``response`` is batched along axis 1 and
        ``sample_locs`` is batched along axis 0. ``inducing_locs`` are kept at their
        full size because they define the latent GP dimensions.

        Parameters
        ----------
        stopper
            Stopping rule for the optimizer. If ``None``, uses
            ``liesel.optim.Stopper(epochs=1000, patience=10, rtol=1e-6)``. Objects
            exposing old ``max_iter``/``epochs`` and ``patience`` attributes are
            accepted for compatibility.
        response_validation
            Optional validation response with shape ``(nobs_validation, nloc)``.
            It may have a different number of rows than the training response, but
            must have the same number of locations. When ``mask_nan_response=True``,
            validation NaNs are masked from this array independently of the training
            NaN pattern.
        optimizer
            Optax gradient transformation. Defaults to ``optax.adam(1e-3)``.
        progress_bar
            Whether to show Liesel's optimization progress bar.
        batch_size
            Number of locations per mini-batch. ``None`` uses all locations in one
            batch. If supplied with ``shuffle_batches=False``, it must divide the
            number of locations exactly because Liesel drops incomplete remainder
            batches. With ``shuffle_batches=True``, arbitrary batch sizes up to the
            number of locations are allowed; a shuffled remainder is omitted in each
            epoch.
        seed
            Random seed used by Liesel for batching and optimizer bookkeeping.
        shuffle_batches
            Whether to shuffle location batches at the start of each epoch. Ignored
            when ``batch_size is None``.
        scale_loss
            Whether Liesel's negative log-probability loss should be divided by the
            training sample size.
        validation_strategy
            Validation objective passed to Liesel. ``"log_lik"`` uses only the
            likelihood; ``"log_prob"`` also includes the log prior.
        train_monitor
            Training-data monitor strategy used when no validation response is
            supplied. Passed through to ``liesel.optim.OptimEngine``.
        save_position_history
            Whether to store parameter positions for every epoch in the optimization
            history.
        progress_n_updates
            Approximate maximum number of progress-bar updates.

        Returns
        -------
        liesel.optim.state.OptimResult
            Optimization result returned by Liesel. The model graph is updated to
            ``result.best_position`` before returning.

        Raises
        ------
        ValueError
            If ``response_validation`` has an incompatible shape, ``batch_size`` is
            invalid, or location batching fails because a location-shaped component
            is fixed at the full number of locations instead of being scalar or
            derived from ``sample_locs``.
        """
        _validate_location_batch_size(batch_size, self.nloc, shuffle_batches)
        _validate_progress_n_updates(progress_n_updates)

        response_name = self.response.name
        sample_locs = self.locs.sample_locs
        sample_locs_name = sample_locs.name

        response_train = {response_name: self.response.value}
        response_validate = {}
        n_validate_response = 0
        if response_validation is not None:
            response_validation = jnp.asarray(response_validation)
            validation_shape = jnp.shape(response_validation)
            if len(validation_shape) != 2:
                raise ValueError(
                    "response_validation must be a two-dimensional array with "
                    "shape (nobs, nloc)."
                )

            nloc_validation = validation_shape[1]
            if nloc_validation != self.nloc:
                raise ValueError(
                    "response_validation must have the same number of locations "
                    f"as the training response; got {nloc_validation} and {self.nloc}."
                )

            response_validate = {response_name: response_validation}
            n_validate_response = validation_shape[0]

        response_split = loptim.PositionSplit(
            Position(response_train),
            Position(response_validate),
            Position({}),
            self.nobs,
            n_validate_response,
            0,
        )

        batch_position_keys = [response_name]
        batch_axes = {response_name: 1}
        split: Any = response_split

        if sample_locs_name in self.graph.vars:
            sample_locs_train = {sample_locs_name: sample_locs.value}
            sample_locs_validate = (
                {sample_locs_name: sample_locs.value}
                if response_validation is not None
                else {}
            )
            n_validate_locs = self.nloc if response_validation is not None else 0
            sample_locs_split = loptim.PositionSplit(
                Position(sample_locs_train),
                Position(sample_locs_validate),
                Position({}),
                self.nloc,
                n_validate_locs,
                0,
            )
            split = loptim.PositionSplitManager([response_split, sample_locs_split])
            batch_position_keys.append(sample_locs_name)
            batch_axes[sample_locs_name] = 0

        location_batches = loptim.Batches(
            batch_position_keys,
            self.nloc,
            batch_size,
            shuffle_batches if batch_size is not None else False,
            batch_axes,
        )
        batches: Any = (
            loptim.BatchManager([location_batches])
            if isinstance(split, loptim.PositionSplitManager)
            else location_batches
        )
        opt = loptim.Optimizer(
            self.parameters,
            optimizer=optimizer if optimizer is not None else optax.adam(1e-3),
        )
        optim = loptim.LieselOptim(
            self.graph,
            split=split,
            batches=batches,
            optimizers=[opt],
            stopper=_coerce_stopper(stopper),
            seed=seed,
            validation_strategy=validation_strategy,
            scale_loss=scale_loss,
            train_monitor=train_monitor,
        )
        engine = optim.build_engine()
        engine.show_progress = progress_bar
        engine.save_position_history = save_position_history
        engine.progress_n_updates = progress_n_updates

        try:
            result = engine.fit()
        except (KeyError, TypeError, ValueError) as error:
            if batch_size is None:
                raise

            raise ValueError(
                "Location-batched Model.fit failed. This usually means that a "
                "location-shaped model component is fixed at the full number of "
                "locations instead of being scalar or derived from sample_locs."
            ) from error

        self.fit_result = result
        self.graph.state = self.graph.update_state(self.fit_result.best_position)

        return result

    def init_dist(
        self,
        samples: dict[str, Array] | None = None,
    ) -> PTMDist:
        if samples is None:
            assert self.response.dist_node is not None
            return cast(HDist, self.response.dist_node).init_base_dist()

        assert samples is not None
        pred = self.graph.predict(samples=cast(Any, samples))
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
        if self.mask_nan_response:
            dist = mask_distribution(dist, y)
        lp = dist.log_prob(y)
        return lp
