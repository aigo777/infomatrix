from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from calibration import Calibrator


class CalibratorTests(unittest.TestCase):
    def test_fit_predict_shape_and_finite(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(120, 6))
        y = np.clip(0.5 + 0.2 * x[:, :2], 0.0, 1.0)

        cal = Calibrator()
        self.assertTrue(cal.fit(x, y, alpha=0.5))

        pred = cal.predict(x[:10])
        self.assertEqual(pred.shape, (10, 2))
        self.assertTrue(np.isfinite(pred).all())
        self.assertTrue((pred >= 0.0).all() and (pred <= 1.0).all())

    def test_save_load_roundtrip(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.normal(size=(90, 6))
        y = np.clip(0.5 + 0.15 * x[:, :2], 0.0, 1.0)
        cal = Calibrator()
        self.assertTrue(cal.fit(x, y, alpha=0.3))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.npz")
            cal.save_npz(model_path)
            loaded = Calibrator.load_npz(model_path)

            pred_a = cal.predict(x[:20])
            pred_b = loaded.predict(x[:20])
            self.assertTrue(np.allclose(pred_a, pred_b, atol=1e-6))

    def test_constant_feature_is_safe(self) -> None:
        rng = np.random.default_rng(99)
        x = rng.normal(size=(80, 6))
        x[:, 3] = 5.0
        y = np.clip(0.5 + 0.1 * x[:, :2], 0.0, 1.0)

        cal = Calibrator()
        self.assertTrue(cal.fit(x, y, alpha=1.0))
        pred = cal.predict(x[:5])
        self.assertTrue(np.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
