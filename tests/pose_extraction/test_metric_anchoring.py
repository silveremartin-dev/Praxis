"""
Unit Tests for Optical-to-Metric Scale Estimator (`tests/pose_extraction/test_metric_anchoring.py`).
"""

import pytest
from src.pose_extraction.metric_anchoring import MetricScaleEstimator


def test_metric_scale_factor():
    # 480 px hand with nominal 184 mm -> ~0.3833 mm/px
    scale = MetricScaleEstimator.estimate_scale_factor(480.0)
    assert pytest.approx(scale, rel=1e-3) == 184.0 / 480.0

    with pytest.raises(ValueError, match="must be positive"):
        MetricScaleEstimator.estimate_scale_factor(0.0)


def test_object_dimensions_and_mass_estimation():
    # Workpiece 100px x 60px with hand 400px (scale = 184/400 = 0.46 mm/px)
    # w = 46.0 mm, h = 27.6 mm, d = 23.0 mm (aspect 0.5)
    # volume = 46 * 27.6 * 23 / 1000 = ~29.2 cm3
    # Steel density = 7.85 g/cm3 -> mass ~ 229g
    res = MetricScaleEstimator.estimate_object_from_hand_anchor(
        hand_length_px=400.0,
        object_bbox_px=(100.0, 60.0),
        material="steel",
        hand_displacement_px=10.0,
        object_displacement_px=12.0,
        dt_seconds=0.0333
    )

    assert res.scale_mm_per_pixel == pytest.approx(0.46, rel=1e-2)
    assert res.object_dimensions_mm["width"] == pytest.approx(46.0, rel=1e-2)
    assert res.object_dimensions_mm["height"] == pytest.approx(27.6, rel=1e-2)
    assert res.estimated_mass_grams > 200.0
    assert res.estimated_mass_grams < 260.0
    assert res.hand_speed_m_s > 0.1
    assert res.object_speed_m_s > 0.1
    assert res.relative_speed_ratio == pytest.approx(1.2, rel=1e-2)
