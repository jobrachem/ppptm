import jax
import jax.numpy as jnp
import numpy as np
import torch
import veccs
from batram.legmods import Data, SimpleTM
from jax.typing import ArrayLike
from scipy import stats

from .model import Model


class CompositeTransformations:
    def __init__(
        self,
        marginal: Model,
        dependence: SimpleTM | None = None,
        use_marginal: bool = True,
    ):
        self.marginal = marginal
        self.dependence = dependence
        self.use_marginal = use_marginal

    @property
    def locs(self) -> jax.Array:
        return self.marginal.locs.ordered.value

    @property
    def y(self) -> jax.Array:
        return self.marginal.response.value

    def init_dependence_model(
        self,
        largest_conditioning_set: int = 30,
        theta_init=None,
        smooth=1.5,
        nug_mult=4.0,
    ) -> SimpleTM:
        tloc = torch.tensor(self.locs)
        nloc = self.locs.shape[0]
        float_dtype = np.dtype(self.y.dtype)
        int_dtype = np.dtype(jnp.array([1]).dtype)

        nn, _ = veccs.preceding_neighbors(
            coordinates=np.asarray(tloc, dtype=float_dtype),
            sequence=np.arange(nloc, dtype=int_dtype),
            num_neighbors=largest_conditioning_set,
        )

        nn = np.asarray(nn, dtype=int_dtype)

        tnn = torch.as_tensor(nn)
        train_hg = self.marginal.hg(self.y) if self.use_marginal else self.y

        train_data = Data.new(tloc, torch.tensor(train_hg), tnn)

        tm = SimpleTM(
            train_data,
            theta_init=theta_init,
            linear=False,
            smooth=smooth,
            nug_mult=nug_mult,
        )

        return tm

    def log_score(self, obs: ArrayLike, logdet_addition: ArrayLike = 0.0) -> jax.Array:
        obs = jnp.asarray(obs)
        if self.use_marginal:
            mdist = self.marginal.init_dist()
            obs_hg, obs_hg_logdet = mdist.transformation_and_logdet(obs)
        else:
            obs_hg = obs
            obs_hg_logdet = jnp.asarray(0.0)

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
        locs_model = self.locs
        z = jax.random.normal(key=key, shape=(n, locs_model.shape[0]))

        if fixed_y is None:
            fixed_zt = torch.tensor([])
        elif self.use_marginal:
            fixed_y = jnp.asarray(fixed_y)
            # marginal model expects data at all locations, so we pad the missing ones
            fixed_size = fixed_y.size
            nloc = self.marginal.locs.ordered.value.shape[0]
            fixed_padded = jnp.zeros(nloc).at[:fixed_size].set(jnp.asarray(fixed_y))

            # marginal model excepts a leading axis
            fy = jnp.expand_dims(fixed_padded, 0)
            fixed_zt_ = self.marginal.hg(fy)[:, :fixed_size]
            # turn into tensor and remove leading axis
            fixed_zt = torch.as_tensor(np.asarray(fixed_zt_).copy()).squeeze(0)
        else:
            fixed_zt = torch.as_tensor(fixed_y)

        with torch.no_grad():
            zt = self.dependence.inverse_map(
                torch.tensor(np.asarray(z)), sample_nugget=True, x_fix=fixed_zt
            )
            zt = zt.numpy()

        y = self.marginal.hgi(zt)

        return y
