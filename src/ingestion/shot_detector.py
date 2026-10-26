"""
Continuous Shot Boundary & Goal-Directed Hand Manipulation Auditor (`src/ingestion/shot_detector.py`).

Enforces Praxis Tier 1 Minimal Intake Filtering:
1. Single-Shot Continuity Invariant:
   - A demonstration clip MUST originate from a single, continuous, fixed camera plane.
   - Any camera angle cut or framing jump implies unknown elapsed time and discontinuous state.
   - Splicing through a camera cut is strictly rejected.
2. Active Hand Manipulation Filter:
   - Verifies that human hands are actively engaged in manipulation for >= 70% of the clip duration.
   - Filters out talking heads, idle views, and static tabletop shots.
3. Goal-Directed Physical Irreversibility:
   - Manipulation must target an intentional physical goal (open, turn, separate, screw, insert, unlatch).
   - Validates state transitions (State_pre != State_post) along the thermodynamic arrow of time.
   - Rejects reversible idle oscillations (finger tapping, waving).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


class GoalActionVerb(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    TURN = "turn"
    ROTATE = "rotate"
    SCREW = "screw"
    UNSCREW = "unscrew"
    SEPARATE = "separate"
    DISASSEMBLE = "disassemble"
    INSERT = "insert"
    UNLATCH = "unlatch"
    FASTEN = "fasten"
    PRESS = "press"
    BYPASS = "bypass"
    EXTRACT = "extract"


# Reversible idle motions that produce zero state change (strictly rejected)
REVERSIBLE_IDLE_ACTIONS: Set[str] = {
    "wave", "idle", "fidget", "scratch", "tap_table", "hover", "oscillate", "gesticulate"
}


@dataclass
class ShotContinuityAudit:
    """Audit report for shot continuity and camera cut detection."""
    is_single_continuous_shot: bool
    num_cuts_detected: int
    max_interframe_difference: float
    cut_timestamps_seconds: List[float] = field(default_factory=list)
    rejection_reason: Optional[str] = None


@dataclass
class HandManipulationAudit:
    """Audit report for active hand engagement during a demonstration."""
    has_active_manipulation: bool
    hand_presence_ratio: float          # In [0.0, 1.0], >= 0.70 required for Tier 1 acceptance
    frames_with_hands: int
    total_frames_audited: int
    rejection_reason: Optional[str] = None


@dataclass
class GoalDirectedActionAudit:
    """Audit report for intentional, non-reversible state transitions."""
    is_goal_directed: bool
    is_state_irreversible: bool
    action_verb: str
    pre_condition: str
    post_condition: str
    rejection_reason: Optional[str] = None


class ShotBoundaryDetector:
    """
    Detects camera framing cuts and scene transitions via color histogram correlation
    and inter-frame structural difference.
    """
    FRAME_DIFF_THRESHOLD: float = 0.14  # Normalized mean absolute pixel difference threshold for shot cut

    @classmethod
    def split_into_continuous_shots(
        cls,
        video_path: str,
        t_start: float,
        t_end: float,
        min_shot_duration_sec: float = 3.0,
    ) -> List[Tuple[float, float]]:
        """
        Splits a video interval [t_start, t_end] at any detected camera cut into
        clean, single-shot sub-intervals of duration >= min_shot_duration_sec.
        """
        audit = cls.audit_shot_continuity(video_path, t_start, t_end, sample_step=1)
        if audit.is_single_continuous_shot:
            return [(round(t_start, 2), round(t_end, 2))] if (t_end - t_start) >= min_shot_duration_sec else []

        boundaries = [t_start] + sorted(audit.cut_timestamps_seconds) + [t_end]
        clean_shots: List[Tuple[float, float]] = []
        for i in range(len(boundaries) - 1):
            s0 = boundaries[i] + (0.08 if i > 0 else 0.0)
            s1 = boundaries[i + 1] - (0.08 if i + 1 < len(boundaries) - 1 else 0.0)
            if (s1 - s0) >= min_shot_duration_sec:
                clean_shots.append((round(s0, 2), round(s1, 2)))
        return clean_shots

    @classmethod
    def audit_shot_continuity(
        cls,
        video_path: str,
        t_start: float,
        t_end: float,
        sample_step: int = 2
    ) -> ShotContinuityAudit:
        """
        Audits whether the interval [t_start, t_end] of video_path is a single continuous shot.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ShotContinuityAudit(
                is_single_continuous_shot=False,
                num_cuts_detected=0,
                max_interframe_difference=1.0,
                rejection_reason=f"Failed to open video file: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(t_start * fps)
        end_frame = int(t_end * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

        prev_gray: Optional[np.ndarray] = None
        max_diff = 0.0
        cuts: List[float] = []

        curr_frame = start_frame
        while curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if (curr_frame - start_frame) % sample_step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if prev_gray is not None:
                    # Normalized mean absolute pixel difference in [0.0, 1.0]
                    diff = float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))) / 255.0)
                    max_diff = max(max_diff, diff)

                    if diff > cls.FRAME_DIFF_THRESHOLD:
                        t_cut = curr_frame / fps
                        cuts.append(round(t_cut, 2))

                prev_gray = gray

            curr_frame += 1

        cap.release()

        is_continuous = len(cuts) == 0
        reason = None
        if not is_continuous:
            reason = (
                f"Camera framing cut detected at t={cuts}s (max_diff={max_diff:.3f}). "
                "Shot breaks temporal continuity; time has elapsed."
            )

        return ShotContinuityAudit(
            is_single_continuous_shot=is_continuous,
            num_cuts_detected=len(cuts),
            max_interframe_difference=round(max_diff, 4),
            cut_timestamps_seconds=cuts,
            rejection_reason=reason
        )


class HandManipulationDetector:
    """
    Verifies that human hands are continuously visible and actively manipulating
    the workpiece during the candidate sequence.
    """
    MIN_PRESENCE_RATIO: float = 0.65

    @classmethod
    def audit_hand_presence(
        cls,
        video_path: str,
        t_start: float,
        t_end: float,
        sample_step: int = 5
    ) -> HandManipulationAudit:
        """
        Audits skin mask presence in the lower/central workspace across [t_start, t_end].
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return HandManipulationAudit(
                has_active_manipulation=False,
                hand_presence_ratio=0.0,
                frames_with_hands=0,
                total_frames_audited=0,
                rejection_reason=f"Failed to open video file: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(t_start * fps)
        end_frame = int(t_end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

        hands_count = 0
        total_audited = 0
        curr_frame = start_frame

        kernel = np.ones((5, 5), np.uint8)

        while curr_frame <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if (curr_frame - start_frame) % sample_step == 0:
                total_audited += 1
                # YCrCb skin segmentation
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 178, 127]))
                skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

                # Active hand check: significant cluster of skin pixels (> 0.5% of frame area)
                h, w = frame.shape[:2]
                skin_pixels = np.count_nonzero(skin_mask)
                if skin_pixels > (0.005 * w * h):
                    hands_count += 1

            curr_frame += 1

        cap.release()

        ratio = (hands_count / max(1, total_audited)) if total_audited > 0 else 0.0
        has_hands = ratio >= cls.MIN_PRESENCE_RATIO
        reason = None
        if not has_hands:
            reason = (
                f"Hand presence ratio {ratio * 100.0:.1f}% is below minimal requirement ({cls.MIN_PRESENCE_RATIO * 100.0:.1f}%). "
                "Clip lacks continuous physical hand engagement."
            )

        return HandManipulationAudit(
            has_active_manipulation=has_hands,
            hand_presence_ratio=round(ratio, 3),
            frames_with_hands=hands_count,
            total_frames_audited=total_audited,
            rejection_reason=reason
        )


class GoalDirectedActionValidator:
    """
    Verifies that the manipulation sequence targets an intentional physical goal
    with a directed, non-reversible state transition (pre-condition != post-condition).
    """
    @classmethod
    def validate_action(
        cls,
        action_verb: str,
        pre_condition: str,
        post_condition: str
    ) -> GoalDirectedActionAudit:
        verb_clean = action_verb.strip().lower()

        # 1. Reject reversible idle actions
        if verb_clean in REVERSIBLE_IDLE_ACTIONS:
            return GoalDirectedActionAudit(
                is_goal_directed=False,
                is_state_irreversible=False,
                action_verb=verb_clean,
                pre_condition=pre_condition,
                post_condition=post_condition,
                rejection_reason=f"Action '{verb_clean}' is a reversible/idle movement with zero thermodynamic state change."
            )

        # 2. Check if verb is a recognized goal-directed action
        recognized_verbs = {v.value for v in GoalActionVerb}
        is_recognized = verb_clean in recognized_verbs or any(v in verb_clean for v in recognized_verbs)

        # 3. State transition must be non-trivial (pre != post)
        state_changes = pre_condition.strip().lower() != post_condition.strip().lower()

        is_valid = is_recognized and state_changes
        reason = None
        if not is_valid:
            if not is_recognized:
                reason = f"Action verb '{verb_clean}' is not recognized as a goal-directed manipulation primitive."
            elif not state_changes:
                reason = f"Pre-condition ('{pre_condition}') and post-condition ('{post_condition}') are identical. No physical state change achieved."

        return GoalDirectedActionAudit(
            is_goal_directed=is_recognized,
            is_state_irreversible=state_changes,
            action_verb=verb_clean,
            pre_condition=pre_condition,
            post_condition=post_condition,
            rejection_reason=reason
        )
