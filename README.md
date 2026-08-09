# Predictive-Process Penalized Transformation Models

[![pre-commit](https://github.com/jobrachem/ppptm/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/jobrachem/ppptm/actions/workflows/pre-commit.yml)
[![pytest](https://github.com/jobrachem/ppptm/actions/workflows/pytest.yml/badge.svg)](https://github.com/jobrachem/ppptm/actions/workflows/pytest.yml)
[![pytest-cov](tests/coverage.svg)](https://github.com/jobrachem/ppptm/actions/workflows/pytest.yml)

`ppptm` fits scalable marginal and dependence transformations for spatial
non-Gaussian fields.

[Documentation](https://jobrachem.github.io/ppptm/)

## Installation

```console
pip install "ppptm @ git+https://github.com/jobrachem/ppptm.git"
```

## Quick start

```python
import jax.numpy as jnp
import ppptm

locs = ppptm.unit_grid_vars(ngrid=2)
y = jnp.arange(8.0).reshape(2, 4)

model = ppptm.Model.new_HG(y, locs, nparam=4)
z = model.hg(y)

assert z.shape == y.shape
assert jnp.allclose(model.hgi(z), y, atol=1e-4)
```

`Model` represents the monotone composition `h(g(y))`: `G` supplies a parametric
marginal transform and `H` supplies spatially varying spline coefficients. Use
`hg()` to move observations to the reference scale and `hgi()` to map simulated
values back. `CompositeTransformations` adds dependence modeling on the transformed
scale.

The package provides:

- parametric-only (`Model.new_G`) and spline-augmented (`Model.new_HG`) models;
- Gaussian, skew-normal, skew-t, gamma, and Weibull marginals;
- constant or predictive-process spatial parameters;
- shared or dimension-specific spatial length scales;
- NaN-aware likelihoods and location mini-batch fitting; and
- Vecchia-style dependence modeling through `CompositeTransformations`.

## Application notebooks and data

- [Americas application](demo/americas.ipynb)
- [Global application](demo/global.ipynb)
- [Application data](https://github.com/jobrachem/ppptm/releases/tag/data)

## Development

Install the development and documentation environments, then run the checks:

```console
uv sync --group dev --group docs
uv run pytest
uv run pytest --doctest-modules src/ppptm
uv run mkdocs build --strict
```

Use `uv run mkdocs serve` for a local documentation preview.

## Provenance

The package implements the scalable composite transformations introduced in:

Brachem, J., Wiemann, P.F.V., & Katzfuss, M. (2026). Data-Efficient Generative
Modeling of Non-Gaussian Global Climate Fields via Scalable Composite
Transformations. *arXiv*. https://doi.org/10.48550/arXiv.2602.23311

## License

`ppptm` is distributed under the terms of the [MIT License](LICENSE).
