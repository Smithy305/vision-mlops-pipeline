"""Tests for drift detection threshold logic and degradation helpers."""

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from monitoring.drift_report import (
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    compute_distribution_shift,
    degrade_image,
    flag_drift,
)


# ---------------------------------------------------------------------------
# Degradation helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def rgb_image() -> Image.Image:
    """Return a simple 128x128 RGB test image."""
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestDegradation:
    def test_gaussian_blur_returns_pil(self, rgb_image):
        """apply_gaussian_blur should return a PIL Image."""
        result = apply_gaussian_blur(rgb_image)
        assert isinstance(result, Image.Image)
        assert result.size == rgb_image.size

    def test_jpeg_compression_returns_pil(self, rgb_image):
        """apply_jpeg_compression should return a PIL Image."""
        result = apply_jpeg_compression(rgb_image, quality=20)
        assert isinstance(result, Image.Image)
        assert result.size == rgb_image.size

    def test_gaussian_noise_returns_pil(self, rgb_image):
        """apply_gaussian_noise should return a PIL Image."""
        result = apply_gaussian_noise(rgb_image, std=30.0)
        assert isinstance(result, Image.Image)
        assert result.size == rgb_image.size

    def test_degrade_image_changes_pixels(self, rgb_image):
        """Degraded image pixel values should differ from the original."""
        degraded = degrade_image(rgb_image)
        arr_orig = np.array(rgb_image)
        arr_deg = np.array(degraded)
        assert not np.array_equal(arr_orig, arr_deg)

    def test_degrade_image_preserves_size(self, rgb_image):
        """Degraded image should have the same spatial dimensions."""
        degraded = degrade_image(rgb_image)
        assert degraded.size == rgb_image.size


# ---------------------------------------------------------------------------
# Drift metrics
# ---------------------------------------------------------------------------

def make_ref_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Create a mock reference DataFrame simulating high-confidence predictions."""
    rng = np.random.default_rng(seed)
    classes = [f"class_{i:02d}" for i in range(10)]
    return pd.DataFrame({
        "predicted_class": rng.choice(classes, n),
        "confidence": rng.uniform(0.7, 0.99, n),
        "top1_label_index": rng.integers(0, 10, n),
        "entropy": rng.uniform(0.1, 0.5, n),
    })


def make_degraded_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    """Create a mock current DataFrame simulating low-confidence degraded predictions."""
    rng = np.random.default_rng(seed)
    classes = [f"class_{i:02d}" for i in range(10)]
    return pd.DataFrame({
        "predicted_class": rng.choice(classes, n),
        "confidence": rng.uniform(0.2, 0.5, n),   # much lower
        "top1_label_index": rng.integers(0, 10, n),
        "entropy": rng.uniform(1.0, 2.5, n),       # much higher
    })


class TestComputeDistributionShift:
    def test_returns_expected_keys(self):
        """compute_distribution_shift must return all required keys."""
        ref_df = make_ref_df()
        cur_df = make_degraded_df()
        metrics = compute_distribution_shift(ref_df, cur_df)
        expected_keys = {
            "confidence_mean_shift",
            "entropy_mean_shift",
            "class_distribution_l1",
            "ref_mean_confidence",
            "cur_mean_confidence",
            "ref_mean_entropy",
            "cur_mean_entropy",
        }
        assert expected_keys == set(metrics.keys())

    def test_identical_dfs_give_zero_shift(self):
        """Identical reference and current should give near-zero shift."""
        ref_df = make_ref_df()
        metrics = compute_distribution_shift(ref_df, ref_df)
        assert metrics["confidence_mean_shift"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["entropy_mean_shift"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["class_distribution_l1"] == pytest.approx(0.0, abs=1e-3)

    def test_large_shift_is_detected(self):
        """Clearly degraded distribution should show large confidence shift."""
        ref_df = make_ref_df()
        cur_df = make_degraded_df()
        metrics = compute_distribution_shift(ref_df, cur_df)
        # Mean confidence drops from ~0.85 to ~0.35 → shift > 0.3
        assert metrics["confidence_mean_shift"] > 0.3


class TestFlagDrift:
    def test_no_drift_on_identical_data(self):
        """No drift should be flagged when reference == current."""
        ref_df = make_ref_df()
        metrics = compute_distribution_shift(ref_df, ref_df)
        detected, reasons = flag_drift(metrics)
        assert detected is False
        assert reasons == []

    def test_drift_detected_on_degraded_data(self):
        """Drift should be flagged when confidence drops significantly."""
        ref_df = make_ref_df()
        cur_df = make_degraded_df()
        metrics = compute_distribution_shift(ref_df, cur_df)
        detected, reasons = flag_drift(
            metrics,
            confidence_threshold=0.10,
            entropy_threshold=0.30,
        )
        assert detected is True
        assert len(reasons) > 0

    def test_custom_thresholds_respected(self):
        """Setting very high thresholds should suppress drift flags."""
        ref_df = make_ref_df()
        cur_df = make_degraded_df()
        metrics = compute_distribution_shift(ref_df, cur_df)
        detected, reasons = flag_drift(
            metrics,
            confidence_threshold=999.0,
            entropy_threshold=999.0,
            distribution_threshold=999.0,
        )
        assert detected is False
        assert reasons == []

    def test_reasons_are_strings(self):
        """All reasons returned by flag_drift should be non-empty strings."""
        ref_df = make_ref_df()
        cur_df = make_degraded_df()
        metrics = compute_distribution_shift(ref_df, cur_df)
        _, reasons = flag_drift(metrics, confidence_threshold=0.01)
        for r in reasons:
            assert isinstance(r, str)
            assert len(r) > 0
