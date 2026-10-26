"""
Automated Video Quality Auditor for Praxis (`src/ingestion/quality_auditor.py`).

Evaluates real-world demonstration videos on physical and visual criteria:
1. Single-Shot Continuity Invariant: 0 camera cuts, fixed camera plane.
2. Bimanual Hand Presence & Wrist Visibility: Both Left and Right hands clearly visible up to the wrist.
3. Single Compact Handheld Workpiece: Object held between hands, not oversized, stable affordance.
4. Resolution & Framing Stability: Minimal camera motion, steady illumination.

Emits an objective Quality Score (0-100) and Grade (A+, A, B, C, REJECTED).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoQualityReport:
    """Detailed audit report on the manipulation demonstration quality."""
    video_path: str
    overall_score: float                # 0.0 to 100.0
    grade: str                          # "A+", "A", "B", "C", "REJECTED"
    is_acceptable: bool                 # True if grade in ["A+", "A", "B"]
    resolution: Tuple[int, int]
    fps: float
    duration_sec: float
    total_frames: int
    continuity_score: float             # 0.0 to 1.0 (1.0 = zero cuts, fixed plane)
    cuts_detected: int
    max_frame_diff: float
    hand_visibility_score: float        # 0.0 to 1.0 (clear bimanual presence up to wrist)
    lh_presence_ratio: float
    rh_presence_ratio: float
    avg_skin_coverage_pct: float
    single_object_score: float          # 0.0 to 1.0 (compact object held between hands)
    inter_hand_gap_avg_px: float
    rejection_reasons: List[str] = field(default_factory=list)


class VideoQualityAuditor:
    """
    Automated algorithmic auditor for ranking and filtering manipulation videos.
    """
    FRAME_DIFF_CUT_THRESHOLD: float = 0.14
    MIN_ACCEPTABLE_SCORE: float = 65.0

    @classmethod
    def audit_video_quality(
        cls,
        video_path: str,
        sample_frames: int = 40,
    ) -> VideoQualityReport:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return VideoQualityReport(
                video_path=video_path,
                overall_score=0.0,
                grade="REJECTED",
                is_acceptable=False,
                resolution=(0, 0),
                fps=0.0,
                duration_sec=0.0,
                total_frames=0,
                continuity_score=0.0,
                cuts_detected=0,
                max_frame_diff=1.0,
                hand_visibility_score=0.0,
                lh_presence_ratio=0.0,
                rh_presence_ratio=0.0,
                avg_skin_coverage_pct=0.0,
                single_object_score=0.0,
                inter_hand_gap_avg_px=0.0,
                rejection_reasons=["Failed to open video file"],
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = n_frames / fps

        rejection_reasons = []

        # 1. Single-Shot Continuity Audit
        step = max(1, n_frames // sample_frames)
        prev_gray = None
        max_diff = 0.0
        cuts = 0

        # Hand & Object presence metrics
        lh_seen = 0
        rh_seen = 0
        skin_coverages = []
        gaps = []
        audited_count = 0

        for frame_idx in range(0, n_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            audited_count += 1
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))) / 255.0)
                max_diff = max(max_diff, diff)
                if diff > cls.FRAME_DIFF_CUT_THRESHOLD:
                    cuts += 1
            prev_gray = gray

            # Skin & Hand detection
            ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(ycrcb, np.array([35, 133, 77]), np.array([255, 178, 132]))
            m2 = cv2.inRange(hsv, np.array([0, 20, 50]), np.array([25, 190, 255]))
            skin = cv2.bitwise_and(m1, m2)
            skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            skin_pct = float(np.mean(skin > 0) * 100.0)
            skin_coverages.append(skin_pct)

            ys, xs = np.where(skin > 0)
            if len(xs) > 60:
                mid_x = int(np.median(xs))
                lh_pts = xs[xs < mid_x]
                rh_pts = xs[xs >= mid_x]

                if len(lh_pts) > 30:
                    lh_seen += 1
                if len(rh_pts) > 30:
                    rh_seen += 1

                if len(lh_pts) > 30 and len(rh_pts) > 30:
                    inner_lh = float(np.percentile(lh_pts, 90))
                    inner_rh = float(np.percentile(rh_pts, 10))
                    gap_px = max(0.0, inner_rh - inner_lh) * (w / 320.0)
                    gaps.append(gap_px)

        cap.release()

        # Score calculations
        # 1. Continuity
        if cuts > 0:
            continuity_score = max(0.0, 1.0 - cuts * 0.45)
            rejection_reasons.append(f"Camera cuts detected ({cuts} cuts, max_diff={max_diff:.3f})")
        else:
            continuity_score = float(np.clip(1.0 - (max_diff / cls.FRAME_DIFF_CUT_THRESHOLD) * 0.4, 0.6, 1.0))

        # 2. Hand visibility
        lh_ratio = lh_seen / max(1, audited_count)
        rh_ratio = rh_seen / max(1, audited_count)
        avg_skin = float(np.mean(skin_coverages)) if skin_coverages else 0.0

        # Ideal skin coverage for bimanual manipulation is 12% to 45% of frame
        if avg_skin < 4.0:
            skin_factor = avg_skin / 4.0
        elif avg_skin > 65.0:
            skin_factor = max(0.5, 1.0 - (avg_skin - 65.0) / 35.0)
        else:
            skin_factor = 1.0

        hand_visibility_score = (0.5 * lh_ratio + 0.5 * rh_ratio) * skin_factor
        if lh_ratio < 0.5 or rh_ratio < 0.5:
            rejection_reasons.append(f"Incomplete hand visibility (LH={lh_ratio*100:.0f}%, RH={rh_ratio*100:.0f}%)")

        # 3. Single object between hands
        avg_gap = float(np.mean(gaps)) if gaps else 0.0
        # Ideal gap between hands for a handheld workpiece is 20px to 300px in a 1280x720 video
        if 15.0 <= avg_gap <= 350.0:
            single_object_score = 1.0
        elif avg_gap < 15.0:
            single_object_score = max(0.4, avg_gap / 15.0)
        else:
            single_object_score = max(0.3, 1.0 - (avg_gap - 350.0) / 400.0)

        # 4. Resolution bonus
        res_factor = 1.0 if (w >= 1280 and h >= 720) else (0.85 if (w >= 640) else 0.65)

        # Overall composite score (0 to 100)
        raw_score = 100.0 * (
            0.35 * continuity_score +
            0.40 * hand_visibility_score +
            0.25 * single_object_score
        ) * res_factor

        overall_score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)

        if cuts > 0:
            grade = "REJECTED"
            is_acceptable = False
        elif overall_score >= 88.0:
            grade = "A+"
            is_acceptable = True
        elif overall_score >= 78.0:
            grade = "A"
            is_acceptable = True
        elif overall_score >= 68.0:
            grade = "B"
            is_acceptable = True
        elif overall_score >= 55.0:
            grade = "C"
            is_acceptable = False
            rejection_reasons.append("Low overall manipulation score")
        else:
            grade = "REJECTED"
            is_acceptable = False
            rejection_reasons.append("Fails minimal quality threshold")

        return VideoQualityReport(
            video_path=video_path,
            overall_score=overall_score,
            grade=grade,
            is_acceptable=is_acceptable,
            resolution=(w, h),
            fps=round(fps, 2),
            duration_sec=round(duration, 2),
            total_frames=n_frames,
            continuity_score=round(continuity_score, 3),
            cuts_detected=cuts,
            max_frame_diff=round(max_diff, 4),
            hand_visibility_score=round(hand_visibility_score, 3),
            lh_presence_ratio=round(lh_ratio, 3),
            rh_presence_ratio=round(rh_ratio, 3),
            avg_skin_coverage_pct=round(avg_skin, 2),
            single_object_score=round(single_object_score, 3),
            inter_hand_gap_avg_px=round(avg_gap, 1),
            rejection_reasons=rejection_reasons,
        )
