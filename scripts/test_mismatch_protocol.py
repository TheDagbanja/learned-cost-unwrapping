"""NumPy-only checks for the paired synthetic observation protocol."""

import unittest

import numpy as np

from scripts.mismatch_protocol import (
    PROTOCOL_VERSION, REGIMES, coherence_field, make_scene_latents, noisy_wrapped,
    snaphu_default_coherence_threshold,
)


class MismatchProtocolTest(unittest.TestCase):
    def test_snaphu_single_look_degeneracy(self):
        self.assertEqual(PROTOCOL_VERSION, 3)
        self.assertGreater(snaphu_default_coherence_threshold(1), 1.0)
        self.assertLess(snaphu_default_coherence_threshold(8), 0.4)

    def test_repeatability_and_pairing(self):
        shape = (48, 64)
        a = make_scene_latents(shape, 42, 12)
        b = make_scene_latents(shape, 42, 12)
        c = make_scene_latents(shape, 42, 13)
        np.testing.assert_array_equal(a.real_noise, b.real_noise)
        self.assertFalse(np.array_equal(a.real_noise, c.real_noise))
        np.testing.assert_array_equal(a.patch_centers, b.patch_centers)
        phi = np.zeros(shape, dtype=np.float32)
        gamma = coherence_field(shape, "mixed", 0.75, a)
        np.testing.assert_array_equal(noisy_wrapped(phi, gamma, a),
                                      noisy_wrapped(phi, gamma, b))

    def test_zero_severity_common_reference(self):
        shape = (48, 64)
        latent = make_scene_latents(shape, 42, 12)
        reference = coherence_field(shape, "homogeneous", 0.0, latent)
        for regime in REGIMES:
            np.testing.assert_array_equal(coherence_field(shape, regime, 0.0, latent),
                                          reference)

    def test_spatial_mismatch_and_valid_range(self):
        shape = (48, 64)
        latent = make_scene_latents(shape, 42, 12)
        homogeneous = coherence_field(shape, "homogeneous", 0.75, latent)
        self.assertEqual(float(homogeneous.max() - homogeneous.min()), 0.0)
        for regime in REGIMES[1:]:
            gamma = coherence_field(shape, regime, 0.75, latent)
            self.assertGreater(float(gamma.std()), 0.0)
            self.assertTrue(np.isfinite(gamma).all())
            self.assertGreaterEqual(float(gamma.min()), 0.02 - 1e-6)
            self.assertLessEqual(float(gamma.max()), 0.98 + 1e-6)


if __name__ == "__main__":
    unittest.main()
