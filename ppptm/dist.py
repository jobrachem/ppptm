import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf


class CustomGEV(tfd.GeneralizedExtremeValue):
    def __init__(
        self,
        loc,
        scale,
        concentration,
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

    def log_prob(self, value):
        gev_log_prob = super().log_prob(value)
        gumbel_log_prob = self.gumbel_dist.log_prob(value)

        log_prob = tf.where(
            tf.equal(self.concentration, 0.0), gumbel_log_prob, gev_log_prob
        )

        log_prob = tf.where(tf.is_inf(log_prob), tf.constant(-1e6), log_prob)

        return log_prob
