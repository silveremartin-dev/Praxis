"""
Praxis Video Ingestion & Shot Boundary Mining Module (`src/ingestion/`).
"""

from src.ingestion.slicer import slice_action_video
from src.ingestion.inspector import inspect_video, extract_clip_frames
from src.ingestion.shot_detector import (
    ShotBoundaryDetector,
    HandManipulationDetector,
    GoalDirectedActionValidator,
    GoalActionVerb,
    ShotContinuityAudit,
    HandManipulationAudit,
    GoalDirectedActionAudit,
)

__all__ = [
    "slice_action_video",
    "inspect_video",
    "extract_clip_frames",
    "ShotBoundaryDetector",
    "HandManipulationDetector",
    "GoalDirectedActionValidator",
    "GoalActionVerb",
    "ShotContinuityAudit",
    "HandManipulationAudit",
    "GoalDirectedActionAudit",
]
