import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf


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
