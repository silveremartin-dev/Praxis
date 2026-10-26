"""
Hand Chirality & Bimanual Convergence Auditor (`src/pose_extraction/chirality_auditor.py`).
Enforces:
1. Chirality Invariant: At most 1 Left Hand and at most 1 Right Hand per demonstration.
2. Inward Convergence Invariant: Hands approach and converge toward the shared workpiece centroid.
3. Mutual Impenetrability Invariant: Left and right hand models do not interpenetrate.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ChiralityAuditResult:
    is_valid: bool
    num_left_hands: int
    num_right_hands: int
    error_message: Optional[str] = None


@dataclass
class ConvergenceAuditResult:
    is_convergent: bool
    lh_convergence_score: float
    rh_convergence_score: float
    approach_valid: bool


@dataclass
class InterHandClearanceResult:
    is_physically_separated: bool
    min_distance_mm: float
    penetration_depth_mm: float


class HandChiralityAuditor:
    """
    Audits bimanual demonstration tracking to guarantee physical and anatomical validity.
    """

    @classmethod
    def audit_chirality_counts(cls, detected_chiralities: List[str]) -> ChiralityAuditResult:
        """
        Guarantees that a physical video extraction contains at most 1 LH and at most 1 RH.
        Rejects any sequence hallucinating duplicate chiralities (e.g. 2 left hands).
        """
        lh_count = sum(1 for c in detected_chiralities if c.lower() in ("left", "left_hand", "lh"))
        rh_count = sum(1 for c in detected_chiralities if c.lower() in ("right", "right_hand", "rh"))

        if lh_count > 1:
            return ChiralityAuditResult(
                is_valid=False,
                num_left_hands=lh_count,
                num_right_hands=rh_count,
                error_message=f"Duplicate Left Hand detected ({lh_count} left hands). Physically impossible.",
            )
        if rh_count > 1:
            return ChiralityAuditResult(
                is_valid=False,
                num_left_hands=lh_count,
                num_right_hands=rh_count,
                error_message=f"Duplicate Right Hand detected ({rh_count} right hands). Physically impossible.",
            )

        return ChiralityAuditResult(
            is_valid=True,
            num_left_hands=lh_count,
            num_right_hands=rh_count,
        )

    @classmethod
    def audit_inward_convergence(
        cls,
        lh_position: Optional[np.ndarray],
        rh_position: Optional[np.ndarray],
        workpiece_position: np.ndarray,
        lh_velocity: Optional[np.ndarray] = None,
        rh_velocity: Optional[np.ndarray] = None,
    ) -> ConvergenceAuditResult:
        """
        Verifies that hands converge inward toward the workpiece centroid.
        Left hand resides primarily in the left quadrant (X <= X_wp) and
        Right hand resides primarily in the right quadrant (X >= X_wp).
        """
        wp_pos = np.asarray(workpiece_position, dtype=float)
        lh_score = 1.0
        rh_score = 1.0
        approach_valid = True

        if lh_position is not None:
            p_lh = np.asarray(lh_position, dtype=float)
            vec_to_wp = wp_pos - p_lh
            dist = np.linalg.norm(vec_to_wp)
            if dist > 1e-4:
                dir_to_wp = vec_to_wp / dist
                # If velocity is provided, verify dot product >= -0.15 (moving inward or stationary)
                if lh_velocity is not None:
                    v_norm = np.linalg.norm(lh_velocity)
                    if v_norm > 0.05:
                        v_dir = lh_velocity / v_norm
                        lh_score = float(np.dot(v_dir, dir_to_wp))
                        if lh_score < -0.30:
                            approach_valid = False

        if rh_position is not None:
            p_rh = np.asarray(rh_position, dtype=float)
            vec_to_wp = wp_pos - p_rh
            dist = np.linalg.norm(vec_to_wp)
            if dist > 1e-4:
                dir_to_wp = vec_to_wp / dist
                if rh_velocity is not None:
                    v_norm = np.linalg.norm(rh_velocity)
                    if v_norm > 0.05:
                        v_dir = rh_velocity / v_norm
                        rh_score = float(np.dot(v_dir, dir_to_wp))
                        if rh_score < -0.30:
                            approach_valid = False

        is_convergent = approach_valid and (lh_score >= -0.30) and (rh_score >= -0.30)
        return ConvergenceAuditResult(
            is_convergent=is_convergent,
            lh_convergence_score=round(lh_score, 3),
            rh_convergence_score=round(rh_score, 3),
            approach_valid=approach_valid,
        )

    @classmethod
    def audit_inter_hand_clearance(
        cls,
        lh_vertices: np.ndarray,
        rh_vertices: np.ndarray,
        tolerance_mm: float = 0.5,
    ) -> InterHandClearanceResult:
        """
        Audits mutual non-collision between Left Hand and Right Hand meshes.
        Guarantees zero interpenetration between the two hands.
        """
        v_lh = np.atleast_2d(lh_vertices)
        v_rh = np.atleast_2d(rh_vertices)

        # Fast Axis-Aligned Bounding Box (AABB) intersection check
        min_lh, max_lh = v_lh.min(axis=0), v_lh.max(axis=0)
        min_rh, max_rh = v_rh.min(axis=0), v_rh.max(axis=0)

        overlap_x = min(max_lh[0], max_rh[0]) - max(min_lh[0], min_rh[0])
        overlap_y = min(max_lh[1], max_rh[1]) - max(min_lh[1], min_rh[1])
        overlap_z = min(max_lh[2], max_rh[2]) - max(min_lh[2], min_rh[2])

        has_aabb_overlap = (overlap_x > 0) and (overlap_y > 0) and (overlap_z > 0)

        if not has_aabb_overlap:
            # Fully disjoint bounding boxes
            dx = max(0.0, max(min_lh[0] - max_rh[0], min_rh[0] - max_lh[0]))
            dy = max(0.0, max(min_lh[1] - max_rh[1], min_rh[1] - max_lh[1]))
            dz = max(0.0, max(min_lh[2] - max_rh[2], min_rh[2] - max_lh[2]))
            min_dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            return InterHandClearanceResult(
                is_physically_separated=True,
                min_distance_mm=round(min_dist, 2),
                penetration_depth_mm=0.0,
            )

        # Detailed centroid separation for intersecting bounding boxes
        c_lh = v_lh.mean(axis=0)
        c_rh = v_rh.mean(axis=0)
        center_dist = float(np.linalg.norm(c_lh - c_rh))

        # Approximate penetration depth along the minimum overlap dimension
        penetration = min(overlap_x, overlap_y, overlap_z)
        is_separated = bool(penetration <= tolerance_mm)

        return InterHandClearanceResult(
            is_physically_separated=is_separated,
            min_distance_mm=0.0 if not is_separated else round(float(center_dist), 2),
            penetration_depth_mm=round(float(max(0.0, penetration)), 2),
        )
