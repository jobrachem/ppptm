from collections.abc import Sequence
from typing import Any, cast

import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax import Array
from jax.typing import ArrayLike
from scipy.spatial.distance import pdist

from .nodes.kernel import GPKernel, KernelFactory
from .nodes.ppvar import ParamPredictiveProcessGP
from .nodes.ppvar_rw import SpatPTMCoef
from .util.locs import LocationVars

_IDENTITY_BIJECTOR = tfb.Identity()
_SOFTPLUS_BIJECTOR = tfb.Softplus()


def _positive_or(value: ArrayLike, fallback: ArrayLike) -> Array:
    """Return ``value`` where it is finite and positive, else ``fallback``."""
    value = jnp.asarray(value)
    return jnp.where(jnp.isfinite(value) & (value > 0), value, fallback)


class G:
    """Build parametric marginal distributions for a transformation model.

    Parameters
    ----------
    y
        Response values, conventionally with shape ``(nobs, nloc)``, used only to
        initialize marginal parameters. Non-finite entries are ignored.
    locs
        Ordered sample and inducing-location variables.
    kernel
        Kernel constructor for spatially varying parameters.
    ard
        Whether spatial kernels use one length scale per coordinate dimension;
        ``False`` uses one shared scalar length scale.
    amplitude_prior, length_scale_prior
        Optional Liesel priors for spatial-kernel hyperparameters.
    salt
        Diagonal covariance jitter for spatial parameters; ``1e-6`` by default.
    hyperparam_bijector
        Transformation enforcing the hyperparameter domain.
    amplitude_start
        Initial spatial-kernel amplitude; ``1.0`` by default. The amplitude is the
        standard deviation of a parameter's spatial variation on the unconstrained
        scale, before its bijector.

    Notes
    -----
    Constant parameters are scalars; names included in a constructor's ``locwise``
    argument become predictive-process GPs with shape ``(nloc,)``. Starting values
    from the data do not depend on ``amplitude_start``; a zero data spread falls
    back to a start of one. Kernel length
    scales start at the minimum pairwise inducing-location distance, so at least two
    distinct inducing locations are required.

    Raises
    ------
    ValueError
        When a distribution constructor cannot find any finite response value for
        initialization.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from ppptm import G, unit_grid_vars
    >>> G(jnp.ones((2, 4)), unit_grid_vars(2)).new_gaussian(locwise=[]).distribution.__name__
    'Normal'

    ```
    """

    def __init__(
        self,
        y: ArrayLike,
        locs: LocationVars,
        kernel: KernelFactory = tfk.MaternFiveHalves,
        ard: bool = False,
        amplitude_prior: lsl.Dist | None = None,
        length_scale_prior: lsl.Dist | None = None,
        salt: float = 1e-6,
        hyperparam_bijector: tfb.Bijector = _SOFTPLUS_BIJECTOR,
        amplitude_start: ArrayLike = 1.0,
    ):
        self.y = jnp.asarray(y)
        self.locs = locs
        self.kernel = kernel
        self.ard = ard
        self.amplitude_prior = amplitude_prior
        self.length_scale_prior = length_scale_prior
        self.salt = salt
        self.hyperparam_bijector = hyperparam_bijector

        self.min_dist = pdist(self.locs.ordered_subset.value, metric="euclidean").min()
        self.amplitude_start = jnp.asarray(amplitude_start, dtype=jnp.ones(1).dtype)

    def _finite_y(self) -> Array:
        y = self.y[jnp.isfinite(self.y)]
        if y.size == 0:
            raise ValueError("Cannot initialize G because y has no finite values.")
        return y

    def _mean_y(self) -> Array:
        return jnp.mean(self._finite_y())

    def _std_y(self) -> Array:
        return jnp.std(self._finite_y())

    def _median_y(self) -> Array:
        return jnp.median(self._finite_y())

    def _var_y(self) -> Array:
        return jnp.var(self._finite_y())

    def new_length_scale(self, name: str) -> lsl.Var:
        """Create a positive spatial-kernel length-scale parameter.

        Parameters
        ----------
        name
            Liesel parameter name.

        Returns
        -------
        length_scale :
            Positive parameter initialized from the minimum inducing-point distance.
            Its shape is scalar, or ``(ndim,)`` when ``ard=True``; pass it to
            ``GPKernel`` when constructing a custom location-wise parameter.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, unit_grid_vars
        >>> scale = G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2), ard=True).new_length_scale("ell")
        >>> assert scale.value.shape == (2,)
        >>> assert jnp.all(scale.value > 0)

        ```
        """
        val = (
            jnp.full((self.locs.ordered.value.shape[-1],), fill_value=self.min_dist)
            if self.ard
            else jnp.array(self.min_dist)
        )
        var_ = lsl.Var.new_param(val, distribution=self.length_scale_prior, name=name)
        var_.transform(self.hyperparam_bijector)
        return var_

    def new_amplitude(self, name: str) -> lsl.Var:
        """Create a positive spatial-kernel amplitude parameter.

        Parameters
        ----------
        name
            Liesel parameter name.

        Returns
        -------
        amplitude :
            Positive scalar parameter initialized from ``amplitude_start``. It can
            be shared by covariance nodes for one spatial marginal parameter.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, unit_grid_vars
        >>> amplitude = G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2), amplitude_start=0.5).new_amplitude("amp")
        >>> assert amplitude.value.shape == ()
        >>> assert jnp.allclose(amplitude.value, 0.5)

        ```
        """
        param = lsl.Var.new_param(
            jnp.array(self.amplitude_start),
            distribution=self.amplitude_prior,
            name=name,
        )
        param.transform(self.hyperparam_bijector)
        return param

    def new_param_const(
        self,
        name: str,
        bijector: tfb.Bijector = _IDENTITY_BIJECTOR,
        init_mean: ArrayLike = 0.0,
        prior: lsl.Dist | None = None,
    ) -> lsl.Var:
        """Create a constant marginal parameter.

        Parameters
        ----------
        name
            Liesel parameter name.
        bijector
            Transformation applied to the parameter.
        init_mean
            Initial scalar value on the constrained scale; ``0.0`` by default.
        prior
            Optional Liesel prior.

        Returns
        -------
        parameter :
            Scalar Liesel parameter suitable as a distribution keyword input.

        Raises
        ------
        ValueError
            If ``init_mean`` is not scalar.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, unit_grid_vars
        >>> parameter = G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2)).new_param_const("offset", init_mean=2.0)
        >>> assert parameter.value.shape == ()
        >>> assert jnp.allclose(parameter.value, 2.0)

        ```
        """
        param = lsl.Var.new_param(jnp.asarray(init_mean), distribution=prior, name=name)
        param.transform(bijector)

        if param.value.size > 1:
            raise ValueError(
                f"Initial mean value for {name} was of "
                f"size {param.value.size}; expected scalar."
            )

        return param

    def new_param_locwise(
        self,
        name: str,
        bijector: tfb.Bijector = _IDENTITY_BIJECTOR,
        init_mean: ArrayLike = 0.0,
    ) -> ParamPredictiveProcessGP:
        """Create a spatially varying marginal parameter.

        Parameters
        ----------
        name
            Base name for the parameter and its kernel hyperparameters.
        bijector
            Transformation applied to the GP value.
        init_mean
            Initial scalar GP mean on the unconstrained scale; ``0.0`` by default.

        Returns
        -------
        parameter :
            Predictive-process Gaussian-process parameter with shape ``(nloc,)``.
            Use it as a location-wise keyword input to a Liesel distribution.

        Raises
        ------
        ValueError
            If ``init_mean`` is not scalar.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, unit_grid_vars
        >>> parameter = G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2)).new_param_locwise("offset")
        >>> assert parameter.value.shape == (4,)
        >>> assert parameter.latent_var.value.shape == (4,)

        ```
        """
        kernel = GPKernel(
            kernel=self.kernel,
            amplitude=self.new_amplitude(name=f"{name}_amplitude"),
            length_scale=self.new_length_scale(name=f"{name}_length_scale"),
        )
        param = ParamPredictiveProcessGP(
            locs=self.locs,
            gp_kernel=kernel,
            name=name,
            bijector=bijector,
            salt=self.salt,
        )
        param.mean.value = jnp.asarray(init_mean)
        if param.mean.value.size > 1:
            raise ValueError(
                f"Initial mean value for {name} was of "
                f"size {param.mean.value.size}; expected scalar."
            )
        return param

    def new_param(
        self,
        name: str,
        locwise: bool,
        bijector: tfb.Bijector = _IDENTITY_BIJECTOR,
        init_mean: ArrayLike = 0.0,
    ) -> ParamPredictiveProcessGP | lsl.Var:
        """Create either a constant or spatial marginal parameter.

        Parameters
        ----------
        name
            Parameter name.
        locwise
            Whether the parameter varies over locations.
        bijector
            Transformation applied to the parameter value.
        init_mean
            Initial scalar value on the constrained scale; ``0.0`` by default.

        Returns
        -------
        parameter :
            Constant Liesel parameter or predictive-process GP parameter.

        Notes
        -----
        For a location-wise parameter, ``init_mean`` is first mapped through the
        inverse bijector so that the realized starting value remains on the requested
        constrained scale.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, ParamPredictiveProcessGP, unit_grid_vars
        >>> factory = G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2))
        >>> assert factory.new_param("a", locwise=False).value.shape == ()
        >>> assert isinstance(factory.new_param("b", locwise=True), ParamPredictiveProcessGP)

        ```
        """
        if locwise:
            init_mean = bijector.inverse(init_mean)
            return self.new_param_locwise(name, bijector, init_mean)
        return self.new_param_const(name, bijector, init_mean, prior=None)

    def new_gaussian(self, locwise: Sequence[str] = ("loc", "scale")) -> lsl.Dist:
        """Build a Gaussian marginal distribution.

        Parameters
        ----------
        locwise
            Names of parameters that vary over locations; both ``loc`` and ``scale``
            are spatial by default. Omitted names produce scalar parameters.

        Returns
        -------
        distribution :
            Liesel normal distribution ready for ``Model``. Its location starts at
            the finite response mean and scale at the finite response standard
            deviation, or one if that is zero; keyword values are scalar or ``(nloc,)`` according to
            ``locwise``.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, unit_grid_vars
        >>> dist = G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2)).new_gaussian(locwise=[])
        >>> assert dist["loc"].value.shape == ()
        >>> assert dist["scale"].value.shape == ()

        ```
        """
        loc = self.new_param("loc", "loc" in locwise, init_mean=self._mean_y())
        scale = self.new_param(
            name="scale",
            locwise="scale" in locwise,
            bijector=tfb.Softplus(),
            init_mean=_positive_or(self._std_y(), 1.0),
        )
        return lsl.Dist(tfd.Normal, loc=loc, scale=scale)

    def new_skewnorm(
        self, locwise: Sequence[str] = ("loc", "scale", "skewness")
    ) -> lsl.Dist:
        """Build a skew-normal marginal distribution.

        Parameters
        ----------
        locwise
            Names of parameters that vary over locations; ``loc``, ``scale``, and
            ``skewness`` are spatial by default.

        Returns
        -------
        distribution :
            Liesel two-piece-normal distribution ready for ``Model``. It adds a
            positive skewness parameter initialized at one, so one gives the
            symmetric normal special case.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, unit_grid_vars
        >>> dist = G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2)).new_skewnorm(locwise=[])
        >>> assert set(dist.kwinputs) == {"loc", "scale", "skewness"}
        >>> assert all(node.value.shape == () for node in dist.kwinputs.values())

        ```
        """
        loc = self.new_param("loc", "loc" in locwise, init_mean=self._mean_y())
        scale = self.new_param(
            name="scale",
            locwise="scale" in locwise,
            bijector=tfb.Softplus(),
            init_mean=_positive_or(self._std_y(), 1.0),
        )
        skewness = self.new_param(
            name="skewness",
            locwise="skewness" in locwise,
            bijector=tfb.Softplus(),
            init_mean=jnp.array(1.0),
        )

        return lsl.Dist(tfd.TwoPieceNormal, loc=loc, scale=scale, skewness=skewness)

    def new_skewt(
        self, locwise: Sequence[str] = ("loc", "scale", "skewness")
    ) -> lsl.Dist:
        """Build a skew-t marginal distribution.

        Parameters
        ----------
        locwise
            Names of parameters that vary over locations. ``loc``, ``scale``, and
            ``skewness`` are spatial by default; ``df`` is scalar unless explicitly
            included.

        Returns
        -------
        distribution :
            Liesel two-piece-Student-t distribution ready for ``Model``. Location
            starts at the median, scale at ``1.4826 * MAD`` (or the standard
            deviation, or one, when the MAD is zero), skewness at one, and
            degrees of freedom at ten, providing a robust heavy-tailed alternative.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, unit_grid_vars
        >>> dist = G(jnp.arange(8.0).reshape(2, 4), unit_grid_vars(2)).new_skewt(locwise=[])
        >>> assert set(dist.kwinputs) == {"loc", "scale", "skewness", "df"}
        >>> assert all(node.value.shape == () for node in dist.kwinputs.values())

        ```
        """
        y = self._finite_y()
        med = self._median_y()

        def mad(z):
            if z.size == 0:
                return jnp.array(1e-7)
            m = jnp.median(z)
            return jnp.median(jnp.abs(z - m))

        # Tied data give a zero MAD; fall back to the standard deviation, then one.
        scale_init = _positive_or(1.4826 * mad(y), _positive_or(jnp.std(y), 1.0))

        loc = self.new_param("loc", "loc" in locwise, init_mean=med)

        scale = self.new_param(
            name="scale",
            locwise="scale" in locwise,
            bijector=tfb.Softplus(),
            init_mean=scale_init,
        )

        skewness = self.new_param(
            name="skewness",
            locwise="skewness" in locwise,
            bijector=tfb.Softplus(),
            init_mean=1.0,
        )

        df = self.new_param(
            name="df",
            locwise="df" in locwise,
            bijector=tfb.Softplus(),
            init_mean=10.0,
        )

        return lsl.Dist(
            tfd.TwoPieceStudentT, loc=loc, scale=scale, skewness=skewness, df=df
        )

    def new_gamma(self, locwise: Sequence[str] = ("concentration", "rate")) -> lsl.Dist:
        """Build a gamma marginal distribution using method-of-moments starts.

        Parameters
        ----------
        locwise
            Names of parameters that vary over locations; concentration and rate are
            spatial by default.

        Returns
        -------
        distribution :
            Liesel gamma distribution ready for positive responses in ``Model``.
            Concentration and rate use method-of-moments starts ``mean**2 / variance``
            and ``mean / variance``.

        Notes
        -----
        Responses should be positive. The factory does not validate support before
        computing starting values. A zero variance is replaced by one, and a start
        that is not finite and positive falls back to one.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, unit_grid_vars
        >>> dist = G(jnp.arange(1.0, 9.0).reshape(2, 4), unit_grid_vars(2)).new_gamma(locwise=[])
        >>> assert jnp.all(dist["concentration"].value > 0)
        >>> assert jnp.all(dist["rate"].value > 0)

        ```
        """
        ymean = self._mean_y()
        yvar = _positive_or(self._var_y(), 1.0)
        concentration_init = _positive_or(ymean**2 / yvar, 1.0)
        rate_init = _positive_or(ymean / yvar, 1.0)

        concentration = self.new_param(
            name="concentration",
            locwise="concentration" in locwise,
            bijector=tfb.Softplus(),
            init_mean=concentration_init,
        )
        rate = self.new_param(
            name="rate",
            locwise="rate" in locwise,
            bijector=tfb.Softplus(),
            init_mean=rate_init,
        )

        return lsl.Dist(tfd.Gamma, concentration=concentration, rate=rate)

    def new_weibull(
        self, locwise: Sequence[str] = ("concentration", "scale")
    ) -> lsl.Dist:
        """Build a Weibull marginal distribution with robust log-scale starts.

        Parameters
        ----------
        locwise
            Names of parameters that vary over locations; concentration and scale
            are spatial by default.

        Returns
        -------
        distribution :
            Liesel Weibull distribution ready for positive responses in ``Model``.
            Starting values use the log-response variance and the Euler-Mascheroni
            constant, with concentration clipped to ``[1e-6, 1e6]``.

        Notes
        -----
        Responses should be positive. Values at or below ``1e-12`` are clipped to
        ``1e-12`` for initialization only; this does not change distribution support.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import G, unit_grid_vars
        >>> dist = G(jnp.arange(1.0, 9.0).reshape(2, 4), unit_grid_vars(2)).new_weibull(locwise=[])
        >>> assert jnp.all(dist["concentration"].value > 0)
        >>> assert jnp.all(dist["scale"].value > 0)

        ```
        """
        eps = 1e-12  # to handle any zeros from rounding
        y_pos = jnp.clip(self._finite_y(), eps, None)
        logy = jnp.log(y_pos)

        gamma = 0.5772156649015329  # Euler–Mascheroni constant
        sy2 = logy.var(ddof=1) if logy.size > 1 else 0.0

        concentration_init = jnp.pi / jnp.sqrt(6.0 * jnp.maximum(sy2, 1e-16))
        concentration_init = jnp.clip(concentration_init, 1e-6, 1e6)

        scale_init = jnp.exp(logy.mean() + gamma / concentration_init)

        concentration = self.new_param(
            name="concentration",
            locwise="concentration" in locwise,
            bijector=tfb.Softplus(),
            init_mean=concentration_init,
        )
        scale = self.new_param(
            name="scale",
            locwise="scale" in locwise,
            bijector=tfb.Softplus(),
            init_mean=scale_init,
        )

        return lsl.Dist(tfd.Weibull, concentration=concentration, scale=scale)


class H:
    """Build spatial transformation coefficients for a transformation model.

    Parameters
    ----------
    locs
        Ordered sample and inducing-location variables.
    nparam
        Number ``D`` of spline transformation coefficients; ``40`` by default and at
        least three for the random-walk construction.
    kernel
        Kernel constructor for the spatial coefficient process.
    ard
        Whether the kernel has a separate length scale per coordinate dimension;
        ``False`` uses one shared scalar length scale.
    locwise_amplitude
        Whether the coefficient-process amplitude varies over locations; ``False``
        uses one shared scalar amplitude.
    amplitude_prior, length_scale_prior
        Optional Liesel priors for kernel hyperparameters.
    hyperparam_bijector
        Transformation enforcing the hyperparameter domain.
    salt
        Diagonal covariance jitter; ``1e-6`` by default.
    amplitude_start
        Initial coefficient-process amplitude; ``0.2`` by default. With
        ``locwise_amplitude=True`` it is also the start of the spatial amplitude's
        level and of its spread.

    Notes
    -----
    The coefficient process combines a spatial predictive-process GP with a
    Brownian-motion random walk across spline coefficients. Kernel length scales
    start at the minimum pairwise inducing-location distance, so at least two
    distinct inducing locations are required.

    Examples
    --------
    ```pycon
    >>> from ppptm import H, unit_grid_vars
    >>> H(unit_grid_vars(2), nparam=3).new_coef().value.shape
    (4, 3)

    ```
    """

    def __init__(
        self,
        locs: LocationVars,
        nparam: int = 40,
        kernel: KernelFactory = tfk.MaternFiveHalves,
        ard: bool = False,
        locwise_amplitude: bool = False,
        amplitude_prior: lsl.Dist | None = None,
        length_scale_prior: lsl.Dist | None = None,
        hyperparam_bijector: tfb.Bijector = _SOFTPLUS_BIJECTOR,
        salt: float = 1e-6,
        amplitude_start: ArrayLike = 0.2,
    ):
        self.locs = locs
        self.kernel = kernel
        self.ard = ard
        self.locwise_amplitude = locwise_amplitude
        self.amplitude_prior = amplitude_prior
        self.length_scale_prior = length_scale_prior
        self.salt = salt
        self.nparam = nparam
        self.hyperparam_bijector = hyperparam_bijector
        self.min_dist = pdist(self.locs.ordered_subset.value, metric="euclidean").min()
        self.amplitude_start = jnp.asarray(amplitude_start, dtype=jnp.ones(1).dtype)

    def new_length_scale(self, name: str) -> lsl.Var:
        """Create a positive spatial-kernel length-scale parameter.

        Parameters
        ----------
        name
            Liesel parameter name.

        Returns
        -------
        length_scale :
            Positive parameter initialized from the minimum inducing-point distance.
            Its shape is scalar or ``(ndim,)`` when ``ard=True`` and it controls the
            spatial correlation range of every coefficient surface.

        Examples
        --------
        ```pycon
        >>> from ppptm import H, unit_grid_vars
        >>> scale = H(unit_grid_vars(2), ard=True).new_length_scale("ell")
        >>> assert scale.value.shape == (2,)
        >>> assert (scale.value > 0).all()

        ```
        """
        val = (
            jnp.full((self.locs.ordered.value.shape[-1],), fill_value=self.min_dist)
            if self.ard
            else jnp.array(self.min_dist)
        )
        length_scale = lsl.Var.new_param(
            val, distribution=self.length_scale_prior, name=name
        )
        length_scale.transform(self.hyperparam_bijector)
        return length_scale

    def new_amplitude_const(self, name: str) -> lsl.Var:
        """Create a positive constant coefficient-process amplitude.

        Parameters
        ----------
        name
            Liesel parameter name.

        Returns
        -------
        amplitude :
            Positive scalar amplitude shared by all locations. Use this when a
            common degree of spatial coefficient variation is adequate.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import H, unit_grid_vars
        >>> amplitude = H(unit_grid_vars(2), amplitude_start=0.3).new_amplitude_const("amp")
        >>> assert amplitude.value.shape == ()
        >>> assert jnp.allclose(amplitude.value, 0.3)

        ```
        """
        param = lsl.Var.new_param(
            self.amplitude_start, distribution=self.amplitude_prior, name=name
        )

        sc = tfb.SoftClip(low=self.hyperparam_bijector.inverse(jnp.array(1e-6)))
        bijector = tfb.Chain([self.hyperparam_bijector, sc])
        # bijector = self.hyperparam_bijector
        param.transform(bijector)
        return param

    def new_amplitude_locwise(self, name: str) -> ParamPredictiveProcessGP:
        """Create a spatially varying coefficient-process amplitude.

        Parameters
        ----------
        name
            Base Liesel parameter name.

        Returns
        -------
        amplitude :
            Positive predictive-process GP with shape ``(nloc,)``. Use it when the
            spline coefficient scale should vary across the spatial domain.

        Notes
        -----
        The value is ``hyperparam_bijector(mean + spread * f)``: ``mean`` sets the
        amplitude's level and starts so that the amplitude starts at
        ``amplitude_start`` everywhere, while ``spread``, the kernel amplitude,
        sets how much it varies over space. If ``amplitude_prior`` is present, it is
        transferred to ``mean`` on its unconstrained scale, so the level
        ``hyperparam_bijector(mean)`` follows that prior; ``spread`` uses the same
        prior.

        Examples
        --------
        ```pycon
        >>> from ppptm import H, unit_grid_vars
        >>> amplitude = H(unit_grid_vars(2)).new_amplitude_locwise("amp")
        >>> assert amplitude.value.shape == (4,)
        >>> assert amplitude.latent_var.value.shape == (4,)

        ```
        """
        kernel = GPKernel(
            kernel=self.kernel,
            amplitude=self.new_amplitude_const(name=f"{name}_amplitude"),
            length_scale=self.new_length_scale(name=f"{name}_length_scale"),
        )
        param = ParamPredictiveProcessGP(
            locs=self.locs,
            gp_kernel=kernel,
            name=name,
            bijector=self.hyperparam_bijector,
            salt=self.salt,
        )
        param.mean.value = self.hyperparam_bijector.inverse(self.amplitude_start)

        amplitude_prior = self.amplitude_prior
        if amplitude_prior is not None:
            sc = tfb.SoftClip(low=self.hyperparam_bijector.inverse(jnp.array(1e-6)))
            chain = tfb.Chain([self.hyperparam_bijector, sc])
            bij = tfb.Invert(chain)

            def distfn(**kwargs):
                d = amplitude_prior.distribution(**kwargs)
                return tfd.TransformedDistribution(d, bij)

            dist = lsl.Dist(
                distribution=distfn,
                _name="",
                _needs_seed=False,
                **cast(dict[str, Any], dict(amplitude_prior.kwinputs)),
            )
            param.mean.dist_node = dist
        return param

    def new_amplitude(self, name: str) -> ParamPredictiveProcessGP | lsl.Var:
        """Create the configured constant or spatial coefficient amplitude.

        Parameters
        ----------
        name
            Base Liesel parameter name.

        Returns
        -------
        amplitude :
            Constant scalar or spatial ``(nloc,)`` amplitude according to
            ``locwise_amplitude``. It plugs directly into the coefficient kernel.

        Examples
        --------
        ```pycon
        >>> from ppptm import H, ParamPredictiveProcessGP, unit_grid_vars
        >>> assert H(unit_grid_vars(2)).new_amplitude("a").value.shape == ()
        >>> assert isinstance(H(unit_grid_vars(2), locwise_amplitude=True).new_amplitude("b"), ParamPredictiveProcessGP)

        ```
        """
        if self.locwise_amplitude:
            return self.new_amplitude_locwise(name)
        return self.new_amplitude_const(name)

    def new_coef(self) -> SpatPTMCoef:
        """Create the spatial random-walk spline coefficient process.

        Returns
        -------
        coefficients :
            Spatial predictive-process coefficients with shape ``(nloc, nparam)``.
            Pass this node as ``coef`` to ``Model.new_HG`` to customize spatial
            smoothness, amplitude structure, or priors.

        Examples
        --------
        ```pycon
        >>> from ppptm import H, Model, unit_grid_vars
        >>> import jax.numpy as jnp
        >>> locs = unit_grid_vars(2)
        >>> coef = H(locs, nparam=4, ard=True).new_coef()
        >>> model = Model.new_HG(jnp.arange(8.0).reshape(2, 4), locs, nparam=4, coef=coef)
        >>> assert model.spatial_coef.value.shape == (4, 4)

        ```
        """
        name = "coef"
        gp_kernel = GPKernel(
            kernel=self.kernel,
            amplitude=self.new_amplitude(f"{name}_amplitude"),
            length_scale=self.new_length_scale(f"{name}_length_scale"),
        )

        coef = SpatPTMCoef(
            locs=self.locs,
            gp_kernel=gp_kernel,
            salt=self.salt,
            nparam=self.nparam,
            name="coef",
        )

        return coef
