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
    """Combine a fitted marginal transformation with a spatial dependence model.

    Parameters
    ----------
    marginal
        Fitted or initialized marginal transformation model.
    dependence
        Optional initialized Batram dependence model. Use
        ``init_dependence_model`` when omitted.
    use_marginal
        Whether to transform observations through ``marginal`` before dependence
        modeling; ``True`` by default.

    Attributes
    ----------
    locs
        Ordered coordinates inherited from the marginal model.
    y
        Response values inherited from the marginal model.

    Notes
    -----
    The composite model separates marginal calibration from spatial dependence:
    ``marginal.hg`` maps each location toward the reference scale and Batram models
    the remaining cross-location structure with a Vecchia approximation.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> from ppptm import CompositeTransformations, Model, unit_grid_vars
    >>> marginal = Model.new_G(jnp.arange(18.0).reshape(2, 9), unit_grid_vars(3))
    >>> composite = CompositeTransformations(marginal)
    >>> assert composite.y.shape == (2, 9)
    >>> assert composite.locs.shape == (9, 2)

    ```
    """

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
        """Ordered coordinates used by the marginal and dependence models.

        The array has shape ``(nloc, ndim)`` and is used to construct Vecchia
        neighbor sets.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import CompositeTransformations, Model, unit_grid_vars
        >>> marginal = Model.new_G(jnp.arange(18.0).reshape(2, 9), unit_grid_vars(3))
        >>> CompositeTransformations(marginal).locs.shape
        (9, 2)

        ```
        """
        return self.marginal.locs.ordered.value

    @property
    def y(self) -> jax.Array:
        """Observed fields from the marginal model.

        The array has shape ``(nobs, nloc)`` and supplies training data when the
        dependence model is initialized.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import CompositeTransformations, Model, unit_grid_vars
        >>> marginal = Model.new_G(jnp.arange(18.0).reshape(2, 9), unit_grid_vars(3))
        >>> CompositeTransformations(marginal).y.shape
        (2, 9)

        ```
        """
        return self.marginal.response.value

    def init_dependence_model(
        self,
        largest_conditioning_set: int = 30,
        theta_init=None,
        smooth=1.5,
        nug_mult=4.0,
    ) -> SimpleTM:
        """Initialize a Batram dependence model on transformed observations.

        Parameters
        ----------
        largest_conditioning_set
            Maximum number of preceding neighbors in the Vecchia approximation;
            ``30`` by default. Larger sets improve dependence fidelity but increase
            memory and compute cost.
        theta_init
            Optional initial dependence-model parameters.
        smooth
            Matérn smoothness setting passed to Batram; ``1.5`` by default.
        nug_mult
            Batram nugget multiplier; ``4.0`` by default.

        Returns
        -------
        dependence :
            Initialized Batram ``SimpleTM``, also stored on this instance for scoring
            and sampling. This method constructs data and parameters; it does not run
            an optimizer.

        Raises
        ------
        ValueError
            If the training fields contain missing values. Batram needs complete
            fields, so a marginal model fitted with ``mask_nan_response=True`` on data
            with missing values cannot be used here.

        Notes
        -----
        Preceding neighbors are computed in the existing maximin location order. If
        ``use_marginal=True``, the training fields are first mapped through
        ``marginal.hg``.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import CompositeTransformations, Model, unit_grid_vars
        >>> marginal = Model.new_G(jnp.arange(27.0).reshape(3, 9), unit_grid_vars(3))
        >>> composite = CompositeTransformations(marginal)
        >>> dependence = composite.init_dependence_model(largest_conditioning_set=3)
        >>> assert dependence is composite.dependence

        ```
        """
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
        missing = np.isnan(np.asarray(train_hg))
        if missing.any():
            raise ValueError(
                "The dependence model cannot use responses with missing values: "
                f"{int(missing.sum())} value(s) in {int(missing.any(axis=1).sum())} "
                "field(s) are NaN. Batram needs complete fields."
            )

        train_data = Data.new(tloc, torch.tensor(train_hg), tnn)

        tm = SimpleTM(
            train_data,
            theta_init=theta_init,
            linear=False,
            smooth=smooth,
            nug_mult=nug_mult,
        )
        self.dependence = tm

        return tm

    def log_score(self, obs: ArrayLike, logdet_addition: ArrayLike = 0.0) -> jax.Array:
        """Compute the average negative log score for observations.

        Parameters
        ----------
        obs
            Observation matrix with shape ``(nobs, nloc)``.
        logdet_addition
            Optional Jacobian log-determinant contribution broadcastable to ``obs``;
            scalar zero by default. Use it when scoring data that underwent an
            additional transformation outside this object.

        Returns
        -------
        score :
            Scalar average negative log score across fields after summing location
            contributions. Lower values indicate better probabilistic predictions.

        Raises
        ------
        AssertionError
            If the dependence model has not been initialized.

        Notes
        -----
        The score adds the standard-normal density, Batram Jacobian, marginal
        Jacobian (when enabled), and ``logdet_addition`` before summing over locations.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from ppptm import CompositeTransformations, Model, unit_grid_vars
        >>> y = jnp.arange(27.0).reshape(3, 9)
        >>> composite = CompositeTransformations(Model.new_G(y, unit_grid_vars(3)))
        >>> _ = composite.init_dependence_model(largest_conditioning_set=3)
        >>> score = composite.log_score(y)
        >>> assert score.shape == ()
        >>> assert jnp.isfinite(score)

        ```
        """
        obs = jnp.asarray(obs)
        assert self.dependence is not None

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
        """Sample fields from the fitted composite model.

        Parameters
        ----------
        key
            JAX random key.
        n
            Number of sampled fields; the result's leading dimension is ``n``.
        fixed_y
            Optional one-dimensional prefix of locations conditioned on during
            sampling. Its length must not exceed ``nloc``.

        Returns
        -------
        samples :
            Array with shape ``(n, nloc)`` on the original response scale, suitable
            for maps, summaries, or predictive scoring.

        Raises
        ------
        AssertionError
            If the dependence model has not been initialized.

        Notes
        -----
        Sampling draws standard-normal innovations, applies Batram's inverse map with
        nugget sampling, then applies ``marginal.hgi``. When conditioning and
        ``use_marginal=True``, the fixed prefix is transformed before inversion.

        ``key`` only seeds the standard-normal innovations. Batram samples the nugget
        with torch's global random state, so the same ``key`` gives different draws
        across calls. Call ``torch.manual_seed`` before ``sample`` for reproducible
        draws.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> from jax.random import key
        >>> from ppptm import CompositeTransformations, Model, unit_grid_vars
        >>> y = jnp.arange(27.0).reshape(3, 9)
        >>> composite = CompositeTransformations(Model.new_G(y, unit_grid_vars(3)))
        >>> _ = composite.init_dependence_model(largest_conditioning_set=3)
        >>> draws = composite.sample(key(7), n=2)
        >>> assert draws.shape == (2, 9)

        ```
        """
        assert self.dependence is not None

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
