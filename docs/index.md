# Predictive-Process Penalized Transformation Models

`ppptm` fits scalable marginal and dependence transformations for spatial
non-Gaussian fields.

## Installation

```bash
pip install "ppptm @ git+https://github.com/jobrachem/ppptm.git"
```

## Quick start

```pycon
>>> import jax.numpy as jnp
>>> import ppptm
>>> locs = ppptm.unit_grid_vars(ngrid=2)
>>> y = jnp.arange(8.0).reshape(2, 4)
>>> model = ppptm.Model.new_HG(y, locs, nparam=4)
>>> z = model.hg(y)
>>> assert z.shape == y.shape
>>> assert jnp.allclose(model.hgi(z), y, atol=1e-4)

```

## Mental model

`Model` represents the monotone composition `h(g(y))`. `G` supplies a parametric
marginal transform, while `H` supplies spatially varying spline coefficients. Use
`model.hg()` to move observations to the reference scale and `model.hgi()` to map
simulated values back. `CompositeTransformations` can then model dependence between
locations on the transformed scale.

## Model families

- `Model.new_G` builds a parametric marginal model without a spline correction.
- `Model.new_HG` adds a spatial monotone spline transformation.
- `CompositeTransformations` adds a Vecchia-style dependence model.

Marginals can be Gaussian, skew-normal, skew-t, gamma, or Weibull, with constant or
predictive-process spatial parameters. Spatial kernels support shared or
dimension-specific length scales, and model fitting supports NaN masking and
location mini-batches.

Continue with the [API reference](api.md), or see the
[application notebooks](https://github.com/jobrachem/ppptm/tree/main/demo).
