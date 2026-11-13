from __future__ import annotations

from collections.abc import Mapping
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
from liesel_ptm import (
    LocScaleTransformationDist,
    OnionKnots,
    TransformationDist,
    onion_dist,
)

from .identity import IdentityTransformationDist, LocScaleIdentityTransformationDist
from .marginals import G, H
from .nodes.ppvar_rw import SpatPTMCoef
from .util.locs import LocationVars

KeyArray = Any
type PTMDist = (
    TransformationDist
    | LocScaleTransformationDist
    | IdentityTransformationDist
    | LocScaleIdentityTransformationDist
)


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
        Whether to center the intermediate spline distribution using its finite-bound
        raw mean; ``False`` by default.
    scaled
        Whether to scale the intermediate spline distribution using its finite-bound
        raw variance; ``False`` by default. Neither flag fixes response moments after
        a nonlinear parametric inverse.
    locscale
        Whether to use affine location-scale normalization. Requires a Normal
        parametric distribution because the reference is standard Normal. Use
        ``False`` for non-Gaussian marginals and their full CDF transformation.
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

        if locscale:
            parametric_dist = g_dist.init_dist()
            if not isinstance(parametric_dist, tfd.Normal):
                raise ValueError(
                    "locscale=True requires an affine G transformation: g_dist must "
                    "construct a Normal distribution for the standard Normal "
                    f"reference. Got {type(parametric_dist).__name__}. "
                    "Set locscale=False to use the full parametric CDF transformation."
                )

        match bspline:
            case "onion":
                partial_dist_class = onion_dist(
                    knots=knots,
                    loc_scale=locscale,
                    parametric_distribution=g_dist.distribution,
                    centered=centered,
                    scaled=scaled,
                )
            case "identity":
                dist_class = (
                    LocScaleIdentityTransformationDist
                    if locscale
                    else IdentityTransformationDist
                )
                partial_dist_class = partial(
                    dist_class,
                    parametric_distribution=g_dist.distribution,
                    centered=centered,
                    scaled=scaled,
                )

        self.partial_dist_class = partial_dist_class
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
        dist = self.distribution(*args, **kwargs)
        if not isinstance(dist, TransformationDist):
            raise TypeError(f"Expected a PTM distribution, got {type(dist).__name__}.")
        return dist

    def init_dist(self) -> tfd.Distribution:
        dist = self.init_base_dist()
        if not self.mask_nan_response:
            return dist

        if self.at is None:
            raise RuntimeError(f"{self!r} cannot derive a NaN mask without `at`.")

        return mask_distribution(dist, self.at.value)


class Model:
    """Penalized transformation model for observations at spatial locations.

    Parameters
    ----------
    y
        Two-dimensional response array with shape ``(nobs, nloc)``. Rows are
        replicate fields and columns correspond to ``locs.sample_locs``.
    locs
        Ordered sample and inducing-location variables for the ``nloc`` locations.
    knots
        One-dimensional spline knot sequence. For the onion spline, its length and
        the coefficient dimension must agree with the ``liesel-ptm`` construction.
    coef
        Optional spatial transformation coefficients. A default process is built
        when omitted. Supply either a scalar identity coefficient or an array-valued
        node with shape ``(nloc, D)``.
    g_dist
        Optional parametric Liesel marginal distribution. A Gaussian marginal is
        built when omitted; in that case ``locscale`` is forced to ``True``.
    to_float32
        Whether to build the Liesel graph with float32 values.
    bspline
        Spline construction: ``"onion"`` by default or ``"identity"``.
    locscale
        Whether to use affine location-scale normalization. Requires a Normal
        parametric distribution because the reference is standard Normal. Use
        ``False`` for non-Gaussian marginals and their full CDF transformation.
    mask_nan_response
        Whether NaN responses are excluded from likelihood evaluation; ``False`` by
        default. NaNs remain in ``response.value`` but contribute zero log density.
    auto_update
        Whether the Liesel graph updates automatically.
    centered, scaled
        Whether to center and scale the intermediate spline distribution using its
        finite-bound raw moments; both default to ``False``. These do not guarantee
        the parametric response mean or variance after a nonlinear parametric inverse.

    Attributes
    ----------
    response
        Observed Liesel response variable.
    graph
        Underlying Liesel model graph.
    coef
        Transformation coefficient node.
    g_dist
        Parametric marginal distribution.

    Raises
    ------
    ValueError
        If response and coefficient nodes describe different numbers of locations,
        if NaN masking is requested with non-finite supplied marginal parameters, or
        if ``locs``, ``g_dist`` or ``coef`` already belong to another model.

    Notes
    -----
    The model represents ``h(g(y))``: ``g`` is a parametric marginal transform and
    ``h`` is an optional monotone spline transform. The graph retains location-wise
    parameters and coefficients, so it can predict transformations at new locations
    from posterior samples.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from ppptm import Model, unit_grid_vars
    >>> model = Model.new_G(jnp.ones((2, 4)), unit_grid_vars(2))
    >>> model.response.value.shape
    (2, 4)

    ```
    """

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
        *,
        centered: bool = False,
        scaled: bool = False,
    ):
        knots = jnp.asarray(knots)
        if g_dist is None:
            g_dist = G(y, locs).new_gaussian()
            locscale = True
        elif mask_nan_response:
            _validate_g_dist_values_are_finite(g_dist)

        if coef is None:
            coef = H(locs, nparam=knots.size - 11).new_coef()

        nobs, nloc2 = jnp.shape(y)
        coef_shape = jnp.shape(coef.value)
        if len(coef_shape) == 0:
            nloc1, D = nloc2, 0
        else:
            nloc1, D = coef_shape

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
            centered=centered,
            scaled=scaled,
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
        try:
            self.graph.replace("stub", self.response)
        except RuntimeError as error:
            if "can only be part of one model" not in str(error):
                raise
            raise ValueError(
                "Some inputs of this model already belong to another Model. Liesel "
                "variables can belong to only one model, so build each Model from "
                "its own locations, e.g. Model.new_HG(y, locs.copy()), and its own "
                "g_dist and coef."
            ) from error

    @classmethod
    def new_HG(
        cls,
        y: ArrayLike,
        locs: LocationVars,
        a: float = -7.0,
        b: float = 7.0,
        nparam: int = 40,
        g_dist: lsl.Dist | None = None,
        coef: SpatPTMCoef | lsl.Var | None = None,
        locscale: bool = False,
        mask_nan_response: bool = False,
        auto_update: bool = True,
        *,
        centered: bool = False,
        scaled: bool = False,
    ) -> Model:
        """Construct a model with an onion-spline transformation.

        Parameters
        ----------
        y
            Response array with shape ``(nobs, nloc)``.
        locs
            Ordered sample and inducing-location variables.
        a, b
            Lower and upper spline-knot bounds; ``-7.0`` and ``7.0`` by default.
        nparam
            Number of spatial transformation coefficients; ``40`` by default and at
            least three.
        g_dist
            Optional parametric marginal distribution.
        coef
            Optional spatial predictive-process coefficients or a fixed Liesel
            variable with shape ``(nloc, nparam)``. Omitted coefficients are built
            with ``H``.
        locscale
            Whether to use affine location-scale normalization. Requires a Normal
            parametric distribution because the reference is standard Normal. Use
            ``False`` for non-Gaussian marginals and their full CDF transformation.
        mask_nan_response
            Whether NaN response entries are masked.
        auto_update
            Whether the Liesel graph updates automatically.
        centered, scaled
            Whether to normalize the intermediate spline distribution using its
            finite-bound raw mean and variance; both default to ``False``. This does
            not fix response moments after a nonlinear parametric inverse.

        Returns
        -------
        model :
            Fully constructed ``h(g(y))`` model with response shape
            ``(nobs, nloc)`` and coefficient shape ``(nloc, nparam)``. Fit it with
            ``fit`` or use its transformation methods directly.

        Raises
        ------
        ValueError
            If the supplied coefficient and response location counts differ.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_HG(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2), nparam=4)
        >>> assert model.response.value.shape == (2, 4)
        >>> assert model.spatial_coef.value.shape == (4, 4)

        ```
        """
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
            centered=centered,
            scaled=scaled,
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
        *,
        centered: bool = False,
        scaled: bool = False,
    ) -> Model:
        """Construct a model with the identity spline and no spatial coefficients.

        Parameters
        ----------
        y
            Response array with shape ``(nobs, nloc)``.
        locs
            Ordered sample and inducing-location variables.
        g_dist
            Optional parametric marginal distribution.
        locscale
            Whether a supplied marginal distribution is treated as location-scale;
            ``False`` by default. When ``g_dist`` is omitted, the default Gaussian
            forces this behavior on. With the standard Normal reference, non-Gaussian
            marginals require ``False`` for their full CDF transformation.
        mask_nan_response
            Whether NaN response entries are masked.
        auto_update
            Whether the Liesel graph updates automatically.
        centered, scaled
            Forwarded normalization flags, both ``False`` by default. They have no
            effect on the identity spline's exact parametric distribution.

        Returns
        -------
        model :
            Model containing only the parametric marginal transformation ``g``. Use
            it when a flexible spline term is unnecessary or as a baseline against
            ``new_HG``. Each model needs its own ``LocationVars``; pass
            ``locs.copy()`` to the second model.

        Raises
        ------
        ValueError
            If the response and location counts are incompatible through supplied
            distribution parameters.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2))
        >>> assert model.response.value.shape == (2, 4)
        >>> assert model.D == 0

        ```
        """
        knots = OnionKnots(a=-1.0, b=1.0, nparam=3)
        coef = lsl.Var.new_value(jnp.array(0.0), name="coef")

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
            centered=centered,
            scaled=scaled,
        )
        return model

    @property
    def parameters(self):
        """List names of trainable Liesel parameters in the model graph.

        The list is useful for inspecting optimizer state or selecting posterior
        samples. A ``new_G`` model has marginal parameters only; ``new_HG`` also has
        coefficient-process parameters.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2))
        >>> assert "loc_latent" in model.parameters

        ```
        """
        return list(self.graph.parameters)

    @property
    def spatial_coef(self) -> SpatPTMCoef:
        """Return the spatial spline coefficient node of an ``h(g(y))`` model.

        The node has shape ``(nloc, D)`` and exposes its kernel hyperparameters and
        latent variables for inspection or prior configuration.

        Raises
        ------
        TypeError
            If this is a ``new_G`` model without spatial coefficients.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_HG(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2), nparam=4)
        >>> model.spatial_coef.value.shape
        (4, 4)

        ```
        """
        if not isinstance(self.coef, SpatPTMCoef):
            raise TypeError("This model does not have spatial coefficients.")
        return self.coef

    def fit(
        self,
        *,
        loss_monitor: loptim.LossMonitor,
        stopper: Any | None = None,
        response_validation: ArrayLike | None = None,
        optimizer: optax.GradientTransformation | None = None,
        progress_bar: bool = False,
        batch_size: int | None = None,
        seed: int | None = None,
        shuffle_batches: bool = True,
        scale_loss: bool = False,
        validation_strategy: Literal["log_lik", "log_prob"] = "log_lik",
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
        loss_monitor
            Loss that drives early stopping, the progress bar and
            ``result.position_min_monitor``; passed to
            ``liesel.optim.LieselOptim``. Use ``"validation"`` with
            ``response_validation``. Without validation data, use
            ``"train_full_data"`` for the exact training loss after each epoch, or
            ``liesel.optim.EmaTrainLossMonitor(effective_window=...)`` for an
            exponential moving average of the pre-update training losses, with the
            span measured in epochs. A span of ``1.0`` is the exact pre-update loss
            without mini-batching.
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
        save_position_history
            Whether to store parameter positions for every epoch in the optimization
            history.
        progress_n_updates
            Approximate maximum number of progress-bar updates over the stopper's
            maximum number of epochs.

        Returns
        -------
        liesel.optim.state.OptimResult
            Optimization result returned by Liesel. The model graph is updated to
            ``result.position_min_monitor``, the position with the lowest monitored
            loss, before returning, so subsequent transformations use the best
            fitted parameters. History retention is controlled by
            ``save_position_history``.

        Raises
        ------
        ValueError
            If ``response_validation`` has an incompatible shape, ``batch_size`` is
            invalid, or location batching fails because a location-shaped component
            is fixed at the full number of locations instead of being scalar or
            derived from ``sample_locs``.
        TypeError
            If ``batch_size`` is not an integer or ``None``.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2))
        >>> model.fit(loss_monitor="train_full_data", batch_size=0)
        Traceback (most recent call last):
        ...
        ValueError: batch_size must be a positive integer or None.

        ```
        """
        _validate_location_batch_size(batch_size, self.nloc, shuffle_batches)
        _validate_progress_n_updates(progress_n_updates)

        # Liesel evaluates losses from the current model state, which must not hold
        # outdated nodes. Without auto_update, nodes stay outdated until updated.
        self.graph.update()

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
        stopper = _coerce_stopper(stopper)
        optim = loptim.LieselOptim(
            self.graph,
            split=split,
            batches=batches,
            optimizers=[opt],
            stopper=stopper,
            seed=seed,
            validation_strategy=validation_strategy,
            scale_loss=scale_loss,
            loss_monitor=loss_monitor,
            save_position_history=save_position_history,
            show_progress=progress_bar,
            progress_update_every=max(1, -(-stopper.epochs // progress_n_updates)),
        )
        engine = optim.build_engine()

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
        self.graph.state = self.graph.update_state(self.fit_result.position_min_monitor)

        return result

    def init_dist(
        self,
        samples: dict[str, Array] | None = None,
        newdata: Mapping[str, ArrayLike] | None = None,
    ) -> PTMDist:
        """Initialize the transformation distribution at model or posterior values.

        Parameters
        ----------
        samples
            Optional posterior parameter samples. Leading sample dimensions are
            preserved in the returned distribution. Values should come from this
            model's graph so their trailing dimensions match model parameters.
            Shared scalar distribution parameters receive a trailing singleton
            location axis, keeping posterior draws separate from locations.
        newdata
            Optional replacement graph inputs; requires ``samples``. To predict at
            new locations, provide ``{"sample_locs": array}`` with shape
            ``(nloc_new, ndim)``.

        Returns
        -------
        distribution :
            Transformation distribution evaluated at model parameters or posterior
            samples. Use it for lower-level transformation, inversion, or density
            operations not exposed by the convenience methods.

        Raises
        ------
        ValueError
            If ``newdata`` is provided without posterior samples.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2))
        >>> distribution = model.init_dist()
        >>> assert hasattr(distribution, "transformation_and_logdet")

        ```
        """
        if samples is None and newdata is not None:
            raise ValueError("newdata requires posterior samples.")
        newdata = dict(newdata) if newdata is not None else None

        if samples is None:
            assert self.response.dist_node is not None
            return cast(HDist, self.response.dist_node).init_base_dist()

        assert self.response.dist_node is not None
        dist_node = cast(HDist, self.response.dist_node)
        sample_shape = self._sample_shape(samples)

        if not sample_shape:
            overlap = set(samples).intersection(newdata or {})
            if overlap:
                raise RuntimeError(
                    f"The following keys are present in both 'samples' and "
                    f"'newdata': {list(overlap)}"
                )

            state = self.graph.update_state(samples | (newdata or {}))
            inputs = {
                name: state[
                    node.var.value_node.name if node.var is not None else node.name
                ].value
                for name, node in dist_node.kwinputs.items()
            }
            return dist_node.partial_dist_class(**inputs)

        inputs = {}
        predict = {}
        fixed = {}

        for input_name, node in dist_node.kwinputs.items():
            parents = self.graph.node_parental_subgraph(node)
            parent_names = {parent.name for parent in parents}
            parent_names.update(
                parent.var.name for parent in parents if parent.var is not None
            )
            target = node.var.name if node.var is not None else node.name

            if parent_names.intersection(samples):
                predict[input_name] = target
            else:
                fixed[input_name] = node

        if fixed:
            submodel = self.graph.parental_submodel(*fixed.values())
            fixed_newdata = {
                name: value
                for name, value in (newdata or {}).items()
                if name in submodel.vars
            }
            state = submodel.update_state(submodel.convert_position(fixed_newdata))

            for input_name, node in fixed.items():
                state_name = (
                    node.var.value_node.name if node.var is not None else node.name
                )
                value = jnp.asarray(state[state_name].value)
                inputs[input_name] = (
                    value
                    if input_name == "coef" and value.ndim == 0
                    else jnp.broadcast_to(value, sample_shape + value.shape)
                )

        if predict:
            pred = self.graph.predict(
                samples=cast(Any, samples),
                predict=list(dict.fromkeys(predict.values())),
                newdata=newdata,
            )
            inputs.update(
                {input_name: pred[target] for input_name, target in predict.items()}
            )

        # Shared scalar parameters still need a location axis after posterior axes.
        # Coefficients retain their (location, basis) axes or scalar identity sentinel.
        for input_name, value in inputs.items():
            if input_name != "coef" and jnp.ndim(value) == len(sample_shape):
                inputs[input_name] = jnp.expand_dims(value, axis=-1)

        return dist_node.partial_dist_class(**inputs)

    def _sample_shape(self, samples: dict[str, Array]) -> tuple[int, ...]:
        for name, value in samples.items():
            if name in self.graph.vars:
                model_value = self.graph.vars[name].value
            elif name in self.graph.nodes:
                model_value = self.graph.nodes[name].value
            else:
                continue
            model_ndim = jnp.ndim(model_value)
            value_ndim = jnp.ndim(value)
            if value_ndim >= model_ndim:
                return jnp.shape(value)[: value_ndim - model_ndim]
        raise ValueError("Could not determine posterior sample shape.")

    def _public_transform(
        self,
        value: ArrayLike,
        samples: dict[str, Array] | None,
        newdata: Mapping[str, ArrayLike] | None,
        operation: Any,
        mask_nan: bool = False,
    ) -> Array:
        dist = self.init_dist(samples, newdata)
        value = jnp.asarray(value)
        posterior_ndims = len(self._sample_shape(samples)) if samples is not None else 0
        if posterior_ndims > 0 and value.ndim > 0:
            value = jnp.reshape(
                value,
                value.shape[:-1] + (1,) * posterior_ndims + value.shape[-1:],
            )
        if mask_nan and self.mask_nan_response:
            dist = mask_distribution(dist, value)
        result = operation(dist, value)
        if posterior_ndims == 0:
            return result

        value_ndims = result.ndim - posterior_ndims - 1
        permutation = (
            tuple(range(value_ndims, value_ndims + posterior_ndims))
            + tuple(range(value_ndims))
            + (result.ndim - 1,)
        )
        return jnp.transpose(result, permutation)

    def g(
        self,
        y: ArrayLike,
        samples: dict[str, Array] | None = None,
        newdata: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Apply the parametric marginal transformation.

        Parameters
        ----------
        y
            Response values with locations along the final axis: scalar, ``(nloc,)``,
            or ``(..., nloc)``.
        samples
            Optional posterior parameter samples with arbitrary leading sample shape.
        newdata
            Optional replacement graph inputs used with ``samples``. Providing it
            without ``samples`` raises ``ValueError``.

        Returns
        -------
        transformed :
            Values on the reference distribution's input scale. Without samples the
            input shape is preserved (a scalar broadcasts to ``(nloc,)``); with
            sample shape ``S``, the result has shape ``S + input.shape``.

        Notes
        -----
        ``g`` isolates the parametric marginal transform, which is useful for
        diagnosing whether non-Gaussian behavior is handled by ``G`` before applying
        the spline component.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2))
        >>> transformed = model.g(jnp.arange(4.0))
        >>> assert transformed.shape == (4,)
        >>> assert jnp.allclose(model.gi(transformed), jnp.arange(4.0), atol=1e-4)

        ```
        """
        return self._public_transform(
            y,
            samples,
            newdata,
            lambda dist, value: dist.transformation_and_logdet_parametric(value)[0],
        )

    def gi(
        self,
        yt: ArrayLike,
        samples: dict[str, Array] | None = None,
        newdata: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Invert the parametric marginal transformation.

        Parameters
        ----------
        yt
            Parametrically transformed values with locations along the final axis:
            scalar, ``(nloc,)``, or ``(..., nloc)``.
        samples
            Optional posterior parameter samples with arbitrary leading sample shape.
        newdata
            Optional replacement graph inputs used with ``samples``. Providing it
            without ``samples`` raises ``ValueError``.

        Returns
        -------
        values :
            Values on the original response scale. Without samples the input shape
            is preserved; posterior sample dimensions are prepended when supplied.

        Notes
        -----
        ``gi`` is the inverse of ``g`` up to numerical tolerance and supports
        simulation from the chosen parametric marginal.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2))
        >>> y = jnp.arange(4.0)
        >>> assert jnp.allclose(model.gi(model.g(y)), y, atol=1e-4)

        ```
        """
        return self._public_transform(
            yt,
            samples,
            newdata,
            lambda dist, value: dist.inverse_transformation_parametric(value),
        )

    def h(
        self,
        yt: ArrayLike,
        samples: dict[str, Array] | None = None,
        newdata: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Apply the spline component of the transformation.

        Parameters
        ----------
        yt
            Parametrically transformed values with locations along the final axis:
            scalar, ``(nloc,)``, or ``(..., nloc)``.
        samples
            Optional posterior parameter samples with arbitrary leading sample shape.
        newdata
            Optional replacement graph inputs used with ``samples``. Providing it
            without ``samples`` raises ``ValueError``.

        Returns
        -------
        transformed :
            Values on the spline's output scale. Without samples the input shape is
            preserved; posterior sample dimensions are prepended when supplied.

        Notes
        -----
        ``h`` isolates the monotone spline transformation and is useful for
        inspecting the nonparametric correction learned beyond the marginal ``g``.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_HG(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2), nparam=4)
        >>> transformed = model.h(jnp.arange(4.0))
        >>> assert transformed.shape == (4,)
        >>> assert jnp.allclose(model.hi(transformed), jnp.arange(4.0), atol=1e-4)

        ```
        """
        return self._public_transform(
            yt,
            samples,
            newdata,
            lambda dist, value: dist.transformation_and_logdet_spline(value)[0],
        )

    def hi(
        self,
        zt: ArrayLike,
        samples: dict[str, Array] | None = None,
        newdata: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Invert the spline component of the transformation.

        Parameters
        ----------
        zt
            Spline-transformed values with locations along the final axis: scalar,
            ``(nloc,)``, or ``(..., nloc)``.
        samples
            Optional posterior parameter samples with arbitrary leading sample shape.
        newdata
            Optional replacement graph inputs used with ``samples``. Providing it
            without ``samples`` raises ``ValueError``.

        Returns
        -------
        values :
            Values on the parametric-transformation scale. Without samples the input
            shape is preserved; posterior sample dimensions are prepended when
            supplied.

        Notes
        -----
        ``hi`` is the numerical inverse of ``h`` and is used during generative
        sampling after drawing on the transformed scale.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_HG(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2), nparam=4)
        >>> yt = jnp.arange(4.0)
        >>> assert jnp.allclose(model.hi(model.h(yt)), yt, atol=1e-4)

        ```
        """
        return self._public_transform(
            zt,
            samples,
            newdata,
            lambda dist, value: dist.inverse_transformation_spline(value),
        )

    def hg(
        self,
        y: ArrayLike,
        samples: dict[str, Array] | None = None,
        newdata: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Apply the full spline and parametric transformation.

        Parameters
        ----------
        y
            Response values with locations along the final axis: scalar, ``(nloc,)``,
            or ``(..., nloc)``.
        samples
            Optional posterior parameter samples with arbitrary leading sample shape.
        newdata
            Optional replacement graph inputs used with ``samples``. Providing it
            without ``samples`` raises ``ValueError``.

        Returns
        -------
        transformed :
            Values on the final reference scale. Without samples the input shape is
            preserved; posterior sample dimensions are prepended when supplied.

        Notes
        -----
        This is the forward composition ``h(g(y))`` used before dependence modeling.
        It is the main diagnostic transform for checking whether fitted marginals are
        close to the reference distribution.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_HG(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2), nparam=4)
        >>> transformed = model.hg(jnp.arange(4.0))
        >>> assert transformed.shape == (4,)
        >>> assert jnp.allclose(model.hgi(transformed), jnp.arange(4.0), atol=1e-4)

        ```
        """
        return self._public_transform(
            y,
            samples,
            newdata,
            lambda dist, value: dist.transformation_and_logdet(value)[0],
        )

    def hgi(
        self,
        zt: ArrayLike,
        samples: dict[str, Array] | None = None,
        newdata: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Invert the full spline and parametric transformation.

        Parameters
        ----------
        zt
            Fully transformed values with locations along the final axis: scalar,
            ``(nloc,)``, or ``(..., nloc)``.
        samples
            Optional posterior parameter samples with arbitrary leading sample shape.
        newdata
            Optional replacement graph inputs used with ``samples``. Providing it
            without ``samples`` raises ``ValueError``.

        Returns
        -------
        values :
            Values on the original response scale. Without samples the input shape
            is preserved; posterior sample dimensions are prepended when supplied.

        Notes
        -----
        ``hgi`` applies ``gi(hi(zt))`` conceptually and is the final step when turning
        transformed Gaussian/dependence-model draws into observations.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> model = Model.new_HG(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2), nparam=4)
        >>> y = jnp.arange(4.0)
        >>> assert jnp.allclose(model.hgi(model.hg(y)), y, atol=1e-4)

        ```
        """
        return self._public_transform(
            zt, samples, newdata, lambda dist, value: dist.inverse_transformation(value)
        )

    def log_prob(
        self,
        y: ArrayLike,
        samples: dict[str, Array] | None = None,
        newdata: Mapping[str, ArrayLike] | None = None,
    ) -> Array:
        """Evaluate model log probabilities, masking configured NaN responses.

        Parameters
        ----------
        y
            Response values with locations along the final axis: scalar, ``(nloc,)``,
            or ``(..., nloc)``.
        samples
            Optional posterior parameter samples with arbitrary leading sample shape.
        newdata
            Optional replacement graph inputs used with ``samples``. Providing it
            without ``samples`` raises ``ValueError``.

        Returns
        -------
        log_probability :
            Per-value log probabilities. Without samples the input shape is preserved
            (a scalar broadcasts to ``(nloc,)``); with sample shape ``S``, the result
            has shape ``S + input.shape``. Sum over observation and location axes for
            a joint conditional log likelihood.

        Notes
        -----
        When ``mask_nan_response=True``, NaN entries are replaced safely for
        evaluation and returned with log probability zero. Other non-finite values
        are not masked.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import Model, unit_grid_vars
        >>> y = jnp.arange(8.0).reshape(2, 4).at[0, 0].set(jnp.nan)
        >>> model = Model.new_G(y, unit_grid_vars(2), mask_nan_response=True)
        >>> log_prob = model.log_prob(y)
        >>> assert log_prob.shape == y.shape
        >>> assert log_prob[0, 0] == 0.0

        ```
        """
        return self._public_transform(
            y,
            samples,
            newdata,
            lambda dist, value: dist.log_prob(value),
            mask_nan=True,
        )
