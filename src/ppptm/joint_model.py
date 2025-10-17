import jax
import jax.numpy as jnp
import numpy as np
import torch
from batram.legmods import SimpleTM
from jax.typing import ArrayLike
from scipy import stats

from .model import Model


class CompositeTransformations:
    def __init__(self, marginal: Model, dependence: SimpleTM):
        self.marginal = marginal
        self.dependence = dependence

    def log_score(self, obs: ArrayLike, logdet_addition: ArrayLike = 0.0) -> jax.Array:
        obs = jnp.asarray(obs)
        mdist = self.marginal.init_dist()
        obs_hg, obs_hg_logdet = mdist.transformation_and_logdet(obs)

        with torch.no_grad():
            z, t_logdet = self.dependence.map_and_logdet(torch.tensor(obs_hg))
            z = z.numpy()
            t_logdet = t_logdet.numpy()

        log_prob = stats.norm.logpdf(z) + obs_hg_logdet + t_logdet
        log_prob = log_prob + jnp.asarray(logdet_addition)
        log_prob = log_prob.sum(axis=1)
        log_score = -log_prob.mean()
        return jnp.asarray(log_score)

    def sample(
        self,
        key: jax.Array,
        n: int,
        fixed_y: jax.typing.ArrayLike | None = None,
    ) -> jax.Array:
        locs_model = self.marginal.locs.ordered.value
        z = jax.random.normal(key=key, shape=(n, locs_model.shape[0]))

        if fixed_y is None:
            fixed_zt = torch.tensor([])
        else:
            fixed_y = jnp.asarray(fixed_y)
            # marginal model expects data at all locations, so we pad the missing ones
            fixed_size = fixed_y.size
            nloc = self.marginal.locs.locs.nloc
            fixed_padded = jnp.zeros(nloc).at[:fixed_size].set(jnp.asarray(fixed_y))

            # marginal model excepts a leading axis
            fy = jnp.expand_dims(fixed_padded, 0)
            fixed_zt_ = self.marginal.hg(fy)[:, :fixed_size]
            # turn into tensor and remove leading axis
            fixed_zt = torch.as_tensor(np.asarray(fixed_zt_).copy()).squeeze(0)

        with torch.no_grad():
            zt = self.dependence.inverse_map(
                torch.tensor(np.asarray(z)), sample_nugget=True, x_fix=fixed_zt
            )
            zt = zt.numpy()

        y = self.marginal.hgi(zt)

        return y
