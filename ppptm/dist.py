from collections.abc import Callable

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel_ptm.dist import TransformationDist
from tensorflow_probability.substrates.jax import tf2jax as tf

Array = jax.Array


class CustomGEV(tfd.GeneralizedExtremeValue):
    """
    Parameters
    ----------
    support_penalty
        Penalty value to use instead of ``NaN`` log probabilities for values outside \
        of the distribution's support.
    extended_support_boundary
        If this is ``extended_support_boundary > 0``, the log probabilities for values \
        that are inside the support by ``extended_support_boundary`` will also be \
        replaced with the penalty value. This can increase numerical stability for \
        evaluating the log probability on values that are *just barely* inside the \
        support.
    scale_penalty
        If ``True`` the penalty is scaled with the number of support violations, such \
        that the total penalty pseudo-log-probabilities sum to ``support_penalty``.
    """

    def __init__(
        self,
        loc,
        scale,
        concentration,
        support_penalty: float = 1e6,
        extend_support_boundary_by: float = 0.1,
        scale_penalty: bool = False,
        validate_args=False,
        allow_nan_stats=True,
        name="CustomGEV",
    ):
        super().__init__(
            loc=loc,
            scale=scale,
            concentration=concentration,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
        )

        # Create a Gumbel distribution (used if concentration == 0)
        self.gumbel_dist = tfd.Gumbel(loc=loc, scale=scale)
        self.support_penalty = support_penalty
        self.extend_support_boundary_by = extend_support_boundary_by
        self.scale_penalty = scale_penalty

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # Explicitly using the _parameter_properties classmethod of the parent class,
        # because otherwise TFP will raise a warning. Since the parameters are exactly
        # the same as for the parent class, this is appropriate.
        return super()._parameter_properties(dtype, num_classes)

    def log_prob(self, value):
        # standardized value for testing support
        # y <= 0 means the value is outside the support of the distribution
        y = 1 + self.concentration * ((value - self.loc) / self.scale)

        y = y - self.extend_support_boundary_by

        # Using the inner-outer jnp.where pattern to obtain working gradients
        # see here: https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where # noqa
        # The values created here are only used to avoid NaNs;
        # they will be deselected by the final call to jnp.where.
        x_gumbel_safe = 0.5 * self.scale + self.loc
        x_gev_safe = -0.5 * self.scale / self.concentration + self.loc

        safe_value = jnp.where(
            y <= 0,
            jnp.where(tf.equal(self.concentration, 0.0), x_gumbel_safe, x_gev_safe),
            value,
        )

        # Explicitly fall back to Gumbel distribution for concentration == 0
        # I think the GEV class gives the wrong support in this case, see this issue:
        # https://github.com/tensorflow/probability/issues/1839
        log_prob = jnp.where(
            tf.equal(self.concentration, 0.0),
            self.gumbel_dist.log_prob(safe_value),
            super().log_prob(safe_value),
        )

        # the jnp.max() is necessary to avoid dividing by zero in one branch of
        # the jnp.where.
        n_support_violations = jnp.max(jnp.array([(y <= 0).sum(), 1.0]))

        if self.scale_penalty:
            penalty = self.support_penalty / n_support_violations
        else:
            penalty = self.support_penalty

        # returns a penalized log prob, with the penalty getting stronger for stronger
        # deviations from the needed support.
        # say, y = -1, then we have -2 * support penalty
        # say, y = -0.1, then we have -1.1 * support penalty
        # The second case has a smaller penalty, which is what we want.
        pseudo_lp = (y - 1.0) * penalty

        lp = jnp.where(y <= 0, pseudo_lp, log_prob)

        return lp



class CensoredTransformationDist(TransformationDist):
    def __init__(
        self,
        knots: Array,
        coef: Array,
        censoring_threshold: float,
        basis_dot_and_deriv_fn: (
            Callable[[Array, Array], tuple[Array, Array]] | None
        ) = None,
        parametric_distribution: type[tfd.Distribution] | None = None,
        reference_distribution: tfd.Distribution | None = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "SurvivalTransformationDist",
        centered: bool = False,
        scaled: bool = False,
        rowwise_dot: bool = True,
        **parametric_distribution_kwargs,
    ):
        super().__init__(
            knots=knots,
            coef=coef,
            basis_dot_and_deriv_fn=basis_dot_and_deriv_fn,
            parametric_distribution=parametric_distribution,
            reference_distribution=reference_distribution,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            centered=centered,
            scaled=scaled,
            rowwise_dot=rowwise_dot,
            **parametric_distribution_kwargs,
        )

        self.censoring_threshold = censoring_threshold

    def _log_prob(self, value: Array):

        log_prob = super()._log_prob(value)
        return jnp.where(value <= self.censoring_threshold, self.cdf(self.censoring_threshold), log_prob)
    
    # def _log_prob(self, value: Array):
    #     uncensored = value[..., 0]
    #     lower = value[..., 1]
    #     upper = value[..., 2]

    #     # Censoring identification based on bounds and transformed_variable
    #     is_right_censored = ~jnp.isnan(lower) & jnp.isnan(upper)
    #     is_left_censored = jnp.isnan(lower) & ~jnp.isnan(upper)
    #     is_interval_censored = ~jnp.isnan(lower) & ~jnp.isnan(upper)
    #     is_uncensored = jnp.isnan(lower) & jnp.isnan(upper)

    #     uncensored_safe = jnp.where(is_uncensored, uncensored, 0.0)
    #     lower_safe = jnp.where(is_right_censored, lower, 0.0)
    #     upper_safe = jnp.where(is_left_censored, upper, 1.0)

    #     # clamping for numerical stability
    #     eps = 1e-12  # A small constant
    #     cdf_lower = jnp.clip(self.cdf(lower_safe), eps, 1.0 - 2 * eps)
    #     cdf_upper = jnp.clip(self.cdf(upper_safe), 2 * eps, 1.0 - eps)

    #     right_censored_log_prob = jnp.log1p(-cdf_lower)
    #     left_censored_log_prob = jnp.log(cdf_upper)
    #     interval_censored_log_prob = jnp.log1p(cdf_upper - cdf_lower - 1.0)
    #     uncensored_log_prob = super()._log_prob(uncensored_safe)

    #     # Combine log-probabilities based on the censoring type
    #     total_log_prob = (
    #         jnp.where(is_right_censored, right_censored_log_prob, 0.0)
    #         + jnp.where(is_left_censored, left_censored_log_prob, 0.0)
    #         + jnp.where(is_interval_censored, interval_censored_log_prob, 0.0)
    #         + jnp.where(is_uncensored, uncensored_log_prob, 0.0)
    #     )

    #     return total_log_prob
    


class LocScaleCensoredTransformationDist(CensoredTransformationDist):
    def __init__(
        self,
        knots: Array,
        coef: Array,
        loc: Array,
        scale: Array,
        censoring_threshold: float,
        basis_dot_and_deriv_fn: (
            Callable[[Array, Array], tuple[Array, Array]] | None
        ) = None,
        validate_args: bool = False,
        allow_nan_stats: bool = True,
        name: str = "LocScaleSurvivalTransformationDist",
        centered: bool = False,
        scaled: bool = False,
        rowwise_dot: bool = True,
    ) -> None:
        super().__init__(
            knots=knots,
            coef=coef,
            parametric_distribution=tfd.Normal,
            reference_distribution=tfd.Normal(loc=0.0, scale=1.0),
            basis_dot_and_deriv_fn=basis_dot_and_deriv_fn,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            loc=jnp.atleast_1d(loc),
            scale=jnp.atleast_1d(scale),
            centered=centered,
            scaled=scaled,
            rowwise_dot=rowwise_dot,
            censoring_threshold=censoring_threshold
        )

    def transformation_and_logdet_parametric(self, value: Array) -> tuple[Array, Array]:
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        transf = (value - self.parametric_distribution.mean()) / sd

        logdet = -jnp.log(sd)

        return transf, logdet

    def inverse_transformation_parametric(self, value: Array) -> Array:
        if self.parametric_distribution is None:
            raise RuntimeError

        sd = self.parametric_distribution.stddev()
        m = self.parametric_distribution.mean()
        y = value * sd + m

        return y
