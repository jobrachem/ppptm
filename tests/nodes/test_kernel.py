import jax.numpy as jnp
import liesel.model as lsl
import tensorflow_probability.substrates.jax.math.psd_kernels as tfk
from jax.random import key, uniform

from ppptm.nodes import kernel as kl

s1 = uniform(key(0), (10,))
s2 = uniform(key(1), (10,))
locs = lsl.Var.new_obs(jnp.c_[s1, s2], name="locs")


class TestGPKernel:
    def test_init(self):
        kernel = kl.GPKernel(x1=locs, x2=locs, kernel=tfk.ExponentiatedQuadratic)
        kernel.update()
        assert not jnp.any(jnp.isnan(kernel.value))
        assert kernel.value.shape == (locs.value.shape[0], locs.value.shape[0])

    def test_init_empty(self):
        kernel = kl.GPKernel(kernel=tfk.ExponentiatedQuadratic)
        kernel.update()
        assert not jnp.any(jnp.isnan(kernel.value))
        assert kernel.value.shape == (
            kernel.x1.value.shape[0],
            kernel.x2.value.shape[0],
        )

    def test_init_with_scalar_amplitude(self):
        amplitude = lsl.Var.new_param(2.0, name="amplitude")
        kernel = kl.GPKernel(
            x1=locs, x2=locs, kernel=tfk.ExponentiatedQuadratic, amplitude=amplitude
        )
        kernel.update()

        assert not jnp.any(jnp.isnan(kernel.value))
        assert kernel.value.shape == (locs.value.shape[0], locs.value.shape[0])

        kernel0 = kl.GPKernel(x1=locs, x2=locs, kernel=tfk.ExponentiatedQuadratic)
        kernel0.update()

        assert jnp.allclose(kernel.value, amplitude.value**2 * kernel0.value)

    def test_init_with_scalar_length_scale(self):
        length_scale = lsl.Var.new_param(2.0, name="length_scale")
        kernel = kl.GPKernel(
            x1=locs,
            x2=locs,
            kernel=tfk.ExponentiatedQuadratic,
            length_scale=length_scale,
        )
        kernel.update()

        assert not jnp.any(jnp.isnan(kernel.value))
        assert kernel.value.shape == (locs.value.shape[0], locs.value.shape[0])

    def test_init_with_array_length_scale(self):
        ell = jnp.array([2.0, 1.0])
        length_scale = lsl.Var.new_param(ell, name="length_scale")
        kernel = kl.GPKernel(
            x1=locs,
            x2=locs,
            kernel=tfk.ExponentiatedQuadratic,
            length_scale=length_scale,
        )
        kernel.update()

        assert not jnp.any(jnp.isnan(kernel.value))
        assert kernel.value.shape == (locs.value.shape[0], locs.value.shape[0])

    def test_init_with_array_amplitude(self):
        amp = jnp.linspace(0.01, 2.0, 10)
        amplitude = lsl.Var.new_param(amp, name="amplitude")
        kernel = kl.GPKernel(
            x1=locs,
            x2=locs,
            kernel=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
        )
        kernel.update()

        assert not jnp.any(jnp.isnan(kernel.value))
        assert kernel.value.shape == (locs.value.shape[0], locs.value.shape[0])

    def test_init_with_array_amplitude_ard(self):
        amp = jnp.linspace(0.01, 2.0, 10)
        ell = jnp.array([2.0, 1.0])
        length_scale = lsl.Var.new_param(ell, name="length_scale")
        amplitude = lsl.Var.new_param(amp, name="amplitude")
        kernel = kl.GPKernel(
            x1=locs,
            x2=locs,
            kernel=tfk.ExponentiatedQuadratic,
            amplitude=amplitude,
            length_scale=length_scale,
        )
        kernel.update()

        assert not jnp.any(jnp.isnan(kernel.value))
        assert kernel.value.shape == (locs.value.shape[0], locs.value.shape[0])


class TestTFK:
    def test_scalar_amplitude(self):
        amp = 1.0
        s = locs.value
        k = tfk.ExponentiatedQuadratic(amplitude=amp)

        K1 = k.matrix(s, s)

        x1, x2 = s, s

        K2 = (
            tfk.ExponentiatedQuadratic(amplitude=amp)
            .apply(x1=x1[None, :], x2=x2[:, None])
            .swapaxes(-1, -2)
        )

        assert jnp.allclose(K1, K2)

    def test_array_amplitude(self):
        """
        .matrix processes array amplitude by batching,
        not by using different amplitude per location.
        """
        amp = jnp.linspace(0.1, 2.0, 10)  # (k,)
        s = locs.value  # (n, n)
        k = tfk.ExponentiatedQuadratic(amplitude=amp)

        K1 = k.matrix(s, s)
        K1.shape  # (k, n, n)

        k = tfk.ExponentiatedQuadratic(amplitude=1.0)

        K2 = k.matrix(s, s)  # (n, n)

        K3 = jnp.outer(amp, amp) * K2  # (n, n)

        assert K3.shape == (s.shape[0], s.shape[0])

        assert not jnp.allclose(K1, K3)  # False

    def test_ard_kernel(self):
        ell = jnp.array([1.0, 2.0])
        k = tfk.FeatureScaled(
            tfk.ExponentiatedQuadratic(
                amplitude=1.0,
                length_scale=1.0,
            ),
            scale_diag=ell,
        )

        s = locs.value  # (n, n)
        K1 = k.matrix(s, s)
        K1.shape  # (n, n)

        k = tfk.ExponentiatedQuadratic(
            amplitude=1.0,
            length_scale=1.0,
        )

        s = locs.value  # (n, n)
        K2 = k.matrix(s / ell, s / ell)

        assert jnp.allclose(K1, K2)
