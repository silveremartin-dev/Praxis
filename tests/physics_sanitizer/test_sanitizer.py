"""
Unit Tests for Automated Physics Sanitizer & Anthropometric Guard (`tests/physics_sanitizer/test_sanitizer.py`).
"""

import numpy as np
import pytest
import trimesh

from src.physics_sanitizer.sdf import sdf_box, sdf_cylinder, sdf_sphere
from src.physics_sanitizer.sanitizer import PhysicsSanitizer, ContactPoint, SanitizationAudit
from src.pose_extraction.hand_model import AnthropometricHandBounds, AnatomicalHandModel


def test_anthropometric_bounds_validation():
    """Verifies that human hand sizes outside [150mm, 250mm] are strictly rejected."""
    # Nominal adult hand (184mm)
    valid, reason = PhysicsSanitizer.validate_anthropometrics(184.0, 82.0)
    assert valid is True
    assert reason is None

    # Too small (< 150mm)
    valid_small, reason_small = PhysicsSanitizer.validate_anthropometrics(120.0, 60.0)
    assert valid_small is False
    assert "below physiological adult minimum" in reason_small

    # Too large (> 250mm, e.g. 50cm unphysical hand)
    valid_giant, reason_giant = PhysicsSanitizer.validate_anthropometrics(500.0, 150.0)
    assert valid_giant is False
    assert "exceeds physiological adult maximum" in reason_giant


def test_anatomical_hand_model_scale_guard():
    """Verifies AnatomicalHandModel raises ValueError when initialized with unphysical scale."""
    # Scale 1.0 (184mm) -> Valid
    hand_normal = AnatomicalHandModel(scale=1.0)
    assert 150.0 <= hand_normal.total_hand_length_mm <= 250.0

    # Scale 3.0 (552mm, ~55cm) -> Must raise ValueError
    with pytest.raises(ValueError, match="Unphysical hand scale"):
        AnatomicalHandModel(scale=3.0)

    # Scale 0.4 (73.6mm, ~7cm) -> Must raise ValueError
    with pytest.raises(ValueError, match="Unphysical hand scale"):
        AnatomicalHandModel(scale=0.4)


def test_anatomical_joint_angle_clamping():
    """Verifies that impossible backwards bending or hyperextension is clamped to physiological limits."""
    # Extreme negative flexion (backward bend -1.0 rad) and extreme forward overflexion (3.5 rad)
    unphysical_curls = (-1.0, 3.5, -0.5)
    clamped = AnthropometricHandBounds.clamp_curl_angles(unphysical_curls)
    
    assert clamped[0] >= AnthropometricHandBounds.MCP_FLEXION_LIMITS[0]
    assert clamped[0] <= AnthropometricHandBounds.MCP_FLEXION_LIMITS[1]
    assert clamped[1] <= AnthropometricHandBounds.PIP_FLEXION_LIMITS[1]
    assert clamped[2] >= AnthropometricHandBounds.DIP_FLEXION_LIMITS[0]


def test_automated_mesh_penetration_audit():
    """Verifies that audit_mesh_penetration detects penetration accurately."""
    sanitizer = PhysicsSanitizer()
    box_he = np.array([20.0, 10.0, 20.0]) # Solid box [-20, 20] x [-10, 10] x [-20, 20]
    box_sdf = lambda pts: sdf_box(pts, box_he)

    # 1. Sphere entirely outside at X = 35mm
    outside_sphere = trimesh.creation.icosphere(radius=5.0)
    outside_sphere.apply_translation([35.0, 0.0, 0.0])
    max_pen, count, _ = sanitizer.audit_mesh_penetration(outside_sphere, box_sdf)
    assert max_pen == 0.0
    assert count == 0

    # 2. Sphere penetrating into the box at X = 22mm (intersects boundary at X=20mm by 3mm)
    penetrating_sphere = trimesh.creation.icosphere(radius=5.0)
    penetrating_sphere.apply_translation([22.0, 0.0, 0.0])
    max_pen_p, count_p, _ = sanitizer.audit_mesh_penetration(penetrating_sphere, box_sdf)
    assert max_pen_p > 2.5
    assert count_p > 0


def test_automated_collision_resolution_projection():
    """
    Verifies that sanitize_mesh projects penetrating vertices back to the solid surface
    so that final max penetration is strictly zero (L_SDF >= 0).
    """
    sanitizer = PhysicsSanitizer(default_contact_margin_mm=0.0)
    box_he = np.array([20.0, 10.0, 20.0])
    box_sdf = lambda pts: sdf_box(pts, box_he)

    # Sphere penetrating 4mm into the box
    sphere = trimesh.creation.icosphere(radius=5.0)
    sphere.apply_translation([21.0, 0.0, 0.0])

    init_pen, init_count, _ = sanitizer.audit_mesh_penetration(sphere, box_sdf)
    assert init_pen > 3.5

    # Run automated sanitizer
    sanitized_sphere, audit = sanitizer.sanitize_mesh(
        sphere, box_sdf, margin_mm=0.0, hand_length_mm=185.0
    )

    assert audit.initial_max_penetration_mm > 3.5
    assert audit.final_max_penetration_mm <= 1e-3
    assert audit.is_physically_valid is True

    # Check resulting mesh has zero penetrating vertices
    fin_pen, fin_count, _ = sanitizer.audit_mesh_penetration(sanitized_sphere, box_sdf)
    assert fin_pen <= 1e-3
    assert fin_count == 0


def test_gross_penetration_rejection():
    """Verifies that gross, unresolvable kinematic hallucinations (> 25mm) trigger automated rejection."""
    sanitizer = PhysicsSanitizer()
    box_he = np.array([50.0, 50.0, 50.0])
    box_sdf = lambda pts: sdf_box(pts, box_he)

    # Mesh placed deeply inside the center of the box (penetration = 50mm)
    deep_mesh = trimesh.creation.icosphere(radius=2.0)
    deep_mesh.apply_translation([0.0, 0.0, 0.0])

    _, audit = sanitizer.sanitize_mesh(deep_mesh, box_sdf, hand_length_mm=185.0)
    assert audit.is_physically_valid is False
    assert "Gross kinematic penetration" in audit.rejection_reason
