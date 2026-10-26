"""
Automated Optical-to-Metric Scale Estimator for Praxis (`src/pose_extraction/metric_anchoring.py`).

Grounds 2D pixel observations into calibrated 3D physical units using the invariant
anthropometric human hand prior:
1. Scale Factor S [mm/pixel] = NominalHandLength (184.0 mm) / ObservedHandPixelSpan.
2. Metric Dimensions: Object width, height, and depth derived from pixel bounding box.
3. Volumetric Mass: Evaluates object mass from metric volume and material bulk density.
4. Relative Kinematics: Computes real metric velocities (m/s) and relative hand-to-workpiece speeds.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from src.dynamics.inverse_dynamics import MATERIAL_DENSITIES_G_CM3


@dataclass
class MetricEstimationResult:
    """Result of metric scale, dimension, mass, and velocity grounding."""
    scale_mm_per_pixel: float
    hand_length_px: float
    hand_length_mm: float
    object_dimensions_mm: Dict[str, float]
    volume_cm3: float
    estimated_mass_grams: float
    material: str
    hand_speed_m_s: float
    object_speed_m_s: float
    relative_speed_ratio: float


class MetricScaleEstimator:
    """
    Infers metric dimensions, mass, and velocities by anchoring optical pixel
    measurements against physiological human hand bounds.
    """
    NOMINAL_HAND_LENGTH_MM: float = 184.0   # Adult 50th percentile (wrist crease to middle fingertip)
    NOMINAL_PALM_BREADTH_MM: float = 82.0   # Breadth across MCP knuckles

    @classmethod
    def estimate_scale_factor(
        cls,
        observed_hand_length_px: float,
        nominal_hand_length_mm: Optional[float] = None
    ) -> float:
        """
        Computes metric scale S [mm/pixel] from detected hand pixel length.
        """
        nominal_len = nominal_hand_length_mm or cls.NOMINAL_HAND_LENGTH_MM
        if observed_hand_length_px <= 0:
            raise ValueError(f"Hand pixel length must be positive, got {observed_hand_length_px}")
        return float(nominal_len / observed_hand_length_px)

    @classmethod
    def estimate_object_from_hand_anchor(
        cls,
        hand_length_px: float,
        object_bbox_px: Tuple[float, float], # (width_px, height_px)
        material: str = "steel",
        depth_aspect_ratio: float = 0.5,     # Assumed depth/width ratio for handheld solids
        hand_displacement_px: float = 0.0,
        object_displacement_px: float = 0.0,
        dt_seconds: float = 0.0333           # 1/30s
    ) -> MetricEstimationResult:
        """
        Anchors 2D visual observations to metric millimeters, volume, mass, and velocities.
        """
        scale_s = cls.estimate_scale_factor(hand_length_px)

        w_px, h_px = object_bbox_px
        w_mm = float(w_px * scale_s)
        h_mm = float(h_px * scale_s)
        d_mm = float(w_mm * depth_aspect_ratio)

        # Volume approximation in cm3
        volume_cm3 = (w_mm * h_mm * d_mm) / 1000.0

        # Density and mass
        mat_key = material.lower().replace(" ", "_")
        density = MATERIAL_DENSITIES_G_CM3.get(mat_key, 2.70) # Default to aluminum/composite
        mass_grams = float(volume_cm3 * density)

        # Velocities
        dt = max(1e-4, dt_seconds)
        hand_v_m_s = float((hand_displacement_px * scale_s / 1000.0) / dt)
        obj_v_m_s = float((object_displacement_px * scale_s / 1000.0) / dt)
        rel_ratio = float(obj_v_m_s / max(1e-4, hand_v_m_s))

        return MetricEstimationResult(
            scale_mm_per_pixel=round(scale_s, 4),
            hand_length_px=round(hand_length_px, 1),
            hand_length_mm=round(cls.NOMINAL_HAND_LENGTH_MM, 1),
            object_dimensions_mm={
                "width": round(w_mm, 2),
                "height": round(h_mm, 2),
                "depth": round(d_mm, 2),
            },
            volume_cm3=round(volume_cm3, 3),
            estimated_mass_grams=round(mass_grams, 2),
            material=mat_key,
            hand_speed_m_s=round(hand_v_m_s, 4),
            object_speed_m_s=round(obj_v_m_s, 4),
            relative_speed_ratio=round(rel_ratio, 3)
        )
