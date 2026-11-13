"""Transformation distributions whose spline is the identity.

``liesel_ptm`` only ships spline-based transformation distributions. ppptm's
parametric-only models (``Model.new_G``) still need the same interface, so these
subclasses keep the parametric transformation and replace the spline with the
identity. The spline object that the parent class requires is a placeholder and is
never evaluated.
"""

from __future__ import annotations

from typing import Any, Literal, cast

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
from liesel_ptm import (
    LocScaleTransformationDist,
    PTMSpline,
    SplineBijector,
    TransformationDist,
)

Array = Any

# Placeholder required by the parent constructor. The coefficient below has one
# entry, which this spline accepts; neither is used for any computation.
_PLACEHOLDER_SPLINE = PTMSpline(jnp.linspace(-3.0, 3.0, 10))


def _placeholder_coef(coef: Array) -> Array:
    coef = jnp.asarray(coef)
    if coef.ndim < 1:
        return jnp.reshape(coef, (1,))
    return coef


def _identity_moment_diagnostic(dtype: Any) -> dict[str, Array]:
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    return {
        "mean": zero,
        "variance": one,
        "mean_reference": zero,
        "variance_reference": one,
        "mean_abs_error": zero,
        "variance_abs_error": zero,
        "mean_rel_error": zero,
        "variance_rel_error": zero,
        "ok": jnp.asarray(True),
    }


class _IdentitySpline(TransformationDist):
    """Replace the spline part of a transformation distribution by the identity."""

    def __init__(self, coef: Array, bspline: Any = None, **kwargs: Any) -> None:
        # ``bspline`` is accepted so that rebuilding from ``self.parameters``
        # (TFP copies, slicing, pytree unflattening) works.
        super().__init__(
            coef=_placeholder_coef(coef),
            bspline=_PLACEHOLDER_SPLINE if bspline is None else bspline,
            **kwargs,
        )

    @property
    def spline_bijector(self) -> SplineBijector:
        """The spline transformation as a TFP bijector: ``tfb.Identity()``.

        Typed as the parent's ``SplineBijector`` to keep the override compatible.
        """
        return cast(SplineBijector, tfb.Identity())

    def _broadcast_value(self, value: Array) -> Array:
        value = jnp.asarray(value, dtype=self.dtype)
        shape = self._spline_output_shape(value)
        return jnp.broadcast_to(value, shape)

    def transformation_and_logdet_spline(self, value: Array) -> tuple[Array, Array]:
        value = self._broadcast_value(value)
        return value, jnp.zeros_like(value)

    def _transformation_spline(self, value: Array) -> Array:
        return self._broadcast_value(value)

    def inverse_transformation_spline(self, value: Array) -> Array:
        return self._broadcast_value(value)

    def transformation_spline_mean(self) -> Array:
        return jnp.asarray(0.0, dtype=self.dtype)

    def transformation_spline_variance(self, mean: Array | None = None) -> Array:
        return jnp.asarray(1.0, dtype=self.dtype)

    def moment_quadrature_diagnostic(
        self,
        reference_order: int | None = None,
        rtol: float = 1e-4,
        atol: float = 1e-5,
        *,
        target: Literal["spline", "response"] = "spline",
    ) -> dict[str, Array]:
        if target == "spline":
            return _identity_moment_diagnostic(self.dtype)
        return super().moment_quadrature_diagnostic(
            reference_order, rtol, atol, target=target
        )

    def _response_moments(self, order: int | None = None) -> tuple[Array, Array]:
        # The response follows the parametric distribution exactly.
        distribution = self.parametric_distribution or self.reference_distribution
        shape = self._batch_shape_tuple()
        return (
            jnp.broadcast_to(distribution.mean(), shape),
            jnp.broadcast_to(distribution.variance(), shape),
        )


class IdentityTransformationDist(_IdentitySpline):
    """
    Transformation distribution with an identity spline.

    Behaves like ``liesel_ptm.TransformationDist`` with the spline replaced by the
    identity, so the response follows ``parametric_distribution`` exactly.
    ``coef`` is accepted for interface compatibility and is ignored; a scalar is
    enough. ``centered`` and ``scaled`` have no effect.
    """


class LocScaleIdentityTransformationDist(_IdentitySpline, LocScaleTransformationDist):
    """
    Location-scale transformation distribution with an identity spline.

    Behaves like ``liesel_ptm.LocScaleTransformationDist`` with the spline replaced
    by the identity, so the response is ``loc + scale * z`` with ``z`` from the
    standard Normal reference. ``coef`` is accepted for interface compatibility and
    is ignored; a scalar is enough. ``centered`` and ``scaled`` have no effect.
    """
