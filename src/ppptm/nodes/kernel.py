from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax import Array

KernelFactory = Callable[..., tfk.PositiveSemidefiniteKernel]


class GPKernel(lsl.Var):
    """Liesel variable that evaluates a Gaussian-process covariance kernel.

    Parameters
    ----------
    kernel
        TensorFlow Probability positive-semidefinite kernel constructor.
    amplitude
        Scalar or Liesel input; ``1.0`` by default. The covariance is multiplied by
        the outer product of amplitudes. One amplitude per location is allowed only
        when ``x1`` and ``x2`` are the same coordinates, or when both are omitted.
    length_scale
        Positive scalar or length-``ndim`` vector controlling correlation range;
        ``1.0`` by default. A vector enables automatic relevance determination.
    x1, x2
        Coordinates for covariance rows and columns. Omitted coordinates use a
        one-row dummy array until replaced through ``new``. Their final feature
        dimensions must match the length-scale dimension.
    name
        Liesel node name.

    Attributes
    ----------
    amplitude
        Covariance-amplitude input node.
    length_scale
        Covariance-length-scale input node.

    Raises
    ------
    ValueError
        If the amplitude is not scalar and ``x1`` and ``x2`` are different
        coordinates. Scale the field instead for a location-wise amplitude, as
        ``H(locwise_amplitude=True)`` does.

    Notes
    -----
    The wrapped TensorFlow Probability kernel is evaluated with unit amplitude and
    feature scaling, then multiplied by ``outer(amplitude, amplitude)``. For the
    covariance of one set of locations with itself, this permits a different
    marginal scale at each location without triggering TFP's batch semantics for
    array-valued amplitudes. A cross-covariance would need the amplitude at both
    sets of locations, which one vector cannot provide. Without coordinates, the
    node only carries hyperparameters for kernels built with ``new``.

    Examples
    --------
    ```pycon
    >>> import jax.numpy as jnp
    >>> import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
    >>> from ppptm import GPKernel
    >>> GPKernel(tfk.ExponentiatedQuadratic, x1=jnp.zeros((2, 1)), x2=jnp.zeros((2, 1))).value.shape
    (2, 2)

    ```
    """

    def __init__(
        self,
        kernel: KernelFactory,
        amplitude: lsl.Var | lsl.Node | float | Array = 1.0,
        length_scale: lsl.Var | lsl.Node | float | Array = 1.0,
        x1: lsl.Var | lsl.Node | Array | None = None,
        x2: lsl.Var | lsl.Node | Array | None = None,
        name: str = "",
    ) -> None:
        if isinstance(length_scale, lsl.Var | lsl.Node):
            lsval = jnp.asarray(length_scale.value)
        else:
            lsval = jnp.asarray(length_scale)

        amplitude_value = (
            amplitude.value if isinstance(amplitude, lsl.Var | lsl.Node) else amplitude
        )
        has_coordinates = x1 is not None or x2 is not None
        if has_coordinates and x1 is not x2 and jnp.ndim(amplitude_value) > 0:
            raise ValueError(
                "GPKernel needs a scalar amplitude for a covariance between different "
                f"coordinates; got amplitude shape {jnp.shape(amplitude_value)}. For "
                "an amplitude that varies over locations, scale the field instead, as "
                "H(locwise_amplitude=True) does."
            )

        if lsval.shape:
            dummy = jnp.ones((1, lsval.shape[-1]))
        else:
            dummy = jnp.ones((1, 2))
        self.x1 = (
            x1
            if isinstance(x1, lsl.Var | lsl.Node)
            else lsl.Value(dummy if x1 is None else x1)
        )
        self.x2 = (
            x2
            if isinstance(x2, lsl.Var | lsl.Node)
            else lsl.Value(dummy if x2 is None else x2)
        )
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
        """Evaluate this kernel at new coordinates.

        Parameters
        ----------
        x1, x2
            Coordinates with shapes ``(nrow, ndim)`` and ``(ncol, ndim)``.
        name
            Name for the new Liesel node.
        amplitude
            Optional replacement amplitude. ``None`` reuses this node's input.
        length_scale
            Optional replacement length scale. ``None`` reuses this node's input.

        Returns
        -------
        kernel :
            New covariance node with shape ``(nrow, ncol)``. Use it to share kernel
            hyperparameters across inducing and prediction covariance matrices.

        Examples
        --------
        ```pycon
        >>> import jax.numpy as jnp
        >>> import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
        >>> from ppptm import GPKernel
        >>> base = GPKernel(tfk.ExponentiatedQuadratic, length_scale=0.5)
        >>> covariance = base.new(jnp.zeros((3, 2)), jnp.ones((2, 2)), name="K")
        >>> assert covariance.value.shape == (3, 2)

        ```
        """
        amplitude = self.amplitude if amplitude is None else amplitude
        length_scale = self.length_scale if length_scale is None else length_scale
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
        """Covariance-amplitude input node shared by derived kernels.

        Examples
        --------
        ```pycon
        >>> import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
        >>> from ppptm import GPKernel
        >>> float(GPKernel(tfk.ExponentiatedQuadratic, amplitude=2.0).amplitude.value)
        2.0

        ```
        """
        return self.value_node["amplitude"]

    @property
    def length_scale(self):
        """Covariance-length-scale input node shared by derived kernels.

        Examples
        --------
        ```pycon
        >>> import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
        >>> from ppptm import GPKernel
        >>> float(GPKernel(tfk.ExponentiatedQuadratic, length_scale=0.5).length_scale.value)
        0.5

        ```
        """
        return self.value_node["length_scale"]
