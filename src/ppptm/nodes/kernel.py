from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax import Array

KernelFactory = Callable[[Array, Array], tfk.PositiveSemidefiniteKernel]
CustomKernelFactory = Callable[..., tfk.PositiveSemidefiniteKernel]


class GPKernel(lsl.Var):
    def __init__(
        self,
        kernel: KernelFactory,
        amplitude: lsl.Var | lsl.Node | float | Array = jnp.array(1.0),
        length_scale: lsl.Var | lsl.Node | float | Array = jnp.array(1.0),
        x1: lsl.Var | lsl.Node | Array | None = None,
        x2: lsl.Var | lsl.Node | Array | None = None,
        name: str = "",
    ) -> None:
        if isinstance(length_scale, lsl.Var | lsl.Node):
            lsval = jnp.asarray(length_scale.value)
        else:
            lsval = jnp.asarray(length_scale)

        if lsval.shape:
            dummy = jnp.ones((1, lsval.shape[-1]))
        else:
            dummy = jnp.ones((1, 2))
        self.x1 = x1 if x1 is not None else lsl.Value(dummy)
        self.x2 = x2 if x2 is not None else lsl.Value(dummy)
        self.kernel = kernel

        def _evaluate_kernel(x1, x2, amplitude, length_scale):
            kernel_ = tfk.FeatureScaled(
                kernel(
                    amplitude=jnp.array(1.0),
                    length_scale=jnp.array(1.0),
                ),
                scale_diag=length_scale,
            )

            return jnp.outer(amplitude, amplitude) * kernel_.matrix(x1, x2)

        calc = lsl.Calc(
            _evaluate_kernel,
            self.x1,
            self.x2,
            amplitude=amplitude,
            length_scale=length_scale,
        ).update()

        super().__init__(calc, name=name)
        self.update()

    def new(
        self,
        x1: lsl.Var | lsl.Node | Array,
        x2: lsl.Var | lsl.Node | Array,
        name: str,
        amplitude: lsl.Var | lsl.Node | Array | None = None,
        length_scale: lsl.Var | lsl.Node | Array | None = None,
    ) -> GPKernel:
        amplitude = amplitude or self.amplitude
        length_scale = length_scale or self.length_scale
        kernel = GPKernel(
            kernel=self.kernel,
            amplitude=amplitude,
            length_scale=length_scale,
            x1=x1,
            x2=x2,
            name=name,
        )
        return kernel

    @property
    def amplitude(self):
        return self.value_node["amplitude"]

    @property
    def length_scale(self):
        return self.value_node["length_scale"]
