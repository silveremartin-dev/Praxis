"""
Generic 3D Pose, Hand Mesh & Parametric Object Extraction Module for Praxis (`src/pose_extraction/`).
"""

from src.pose_extraction.hand_model import (
    AnatomicalHandModel,
    DexterousRobotHandModel,
    create_smooth_capsule,
    solve_two_bone_ik,
)
from src.pose_extraction.cad_primitives import (
    ParametricCADBuilder,
    create_box,
    create_cylinder,
)
from src.pose_extraction.extractor import (
    FrameVisionFeatures,
    VideoPoseExtractor,
)

# For backwards compatibility
create_smooth_segment = create_smooth_capsule

__all__ = [
    "AnatomicalHandModel",
    "DexterousRobotHandModel",
    "create_smooth_capsule",
    "create_smooth_segment",
    "solve_two_bone_ik",
    "ParametricCADBuilder",
    "create_box",
    "create_cylinder",
    "FrameVisionFeatures",
    "VideoPoseExtractor",
]
