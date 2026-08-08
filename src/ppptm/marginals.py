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


class G:
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
        val = (
            jnp.full((self.locs.ordered.value.shape[-1],), fill_value=self.min_dist)
            if self.ard
            else jnp.array(self.min_dist)
        )
        var_ = lsl.Var.new_param(val, distribution=self.length_scale_prior, name=name)
        var_.transform(self.hyperparam_bijector)
        return var_

    def new_amplitude(self, name: str) -> lsl.Var:
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
        if locwise:
            init_mean = bijector.inverse(init_mean)
            return self.new_param_locwise(name, bijector, init_mean)
        return self.new_param_const(name, bijector, init_mean, prior=None)

    def new_gaussian(self, locwise: Sequence[str] = ("loc", "scale")) -> lsl.Dist:
        loc = self.new_param("loc", "loc" in locwise, init_mean=self._mean_y())
        scale = self.new_param(
            name="scale",
            locwise="scale" in locwise,
            bijector=tfb.Softplus(),
            init_mean=self._std_y(),
        )
        return lsl.Dist(tfd.Normal, loc=loc, scale=scale)

    def new_skewnorm(
        self, locwise: Sequence[str] = ("loc", "scale", "skewness")
    ) -> lsl.Dist:
        loc = self.new_param("loc", "loc" in locwise, init_mean=self._mean_y())
        scale = self.new_param(
            name="scale",
            locwise="scale" in locwise,
            bijector=tfb.Softplus(),
            init_mean=self._std_y(),
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
        y = self._finite_y()
        med = self._median_y()

        def mad(z):
            if z.size == 0:
                return jnp.array(1e-7)
            m = jnp.median(z)
            return jnp.median(jnp.abs(z - m))

        scale_init = 1.4826 * mad(y)

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
        ymean = self._mean_y()
        yvar = self._var_y()
        concentration_init = ymean**2 / yvar
        rate_init = ymean / yvar

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
        param = lsl.Var.new_param(
            self.amplitude_start, distribution=self.amplitude_prior, name=name
        )

        sc = tfb.SoftClip(low=self.hyperparam_bijector.inverse(jnp.array(1e-6)))
        bijector = tfb.Chain([self.hyperparam_bijector, sc])
        # bijector = self.hyperparam_bijector
        param.transform(bijector)
        return param

    def new_amplitude_locwise(self, name: str) -> ParamPredictiveProcessGP:
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
        if self.locwise_amplitude:
            return self.new_amplitude_locwise(name)
        return self.new_amplitude_const(name)

    def new_coef(self) -> SpatPTMCoef:
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
