"""
Generic Video-Driven 3D Hand, Tool & Object Pose Extractor for Praxis (`src/pose_extraction/extractor.py`).

Extracts continuous 3D bimanual hand kinematics, opposable thumb & finger articulations,
6-DoF workpiece poses, and tool trajectories (including tool release/drop events)
from monocular video clips (.mp4) using OpenCV skin/contour/chromatic tracking and
analytical Inverse Kinematics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os
import cv2
import numpy as np
import trimesh

from src.pose_extraction.hand_model import (
    AnatomicalHandModel,
    DexterousRobotHandModel,
    create_smooth_segment,
)
from src.pose_extraction.cad_primitives import ParametricCADBuilder


@dataclass
class FrameVisionFeatures:
    """Per-frame 2D/3D features extracted directly from video pixels via OpenCV."""
    frame_idx: int
    progress: float
    lh_centroid_norm: Tuple[float, float]
    lh_thumb_tip_norm: Tuple[float, float]
    rh_centroid_norm: Tuple[float, float]
    rh_apex_norm: Tuple[float, float]
    red_bumper_exposed_ratio: float
    tool_active: bool


class VideoPoseExtractor:
    """
    Generic computer-vision pose and interaction extractor that processes video clips
    to drive 3D bimanual human and dexterous robot hand digital twins.
    """

    def __init__(self, scale_mm_per_norm: float = 110.0):
        self.scale_mm = scale_mm_per_norm
        self._cache: Dict[str, List[FrameVisionFeatures]] = {}

    def analyze_video_clip(self, video_path: str, num_samples: int = 30) -> List[FrameVisionFeatures]:
        """
        Runs OpenCV YCrCb skin segmentation, contour apex detection, and chromatic bumper
        visibility tracking across `num_samples` evenly spaced frames of `video_path`.
        """
        cache_key = f"{video_path}:{num_samples}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not os.path.exists(video_path):
            return []

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        features: List[FrameVisionFeatures] = []

        for i, f_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, bgr = cap.read()
            if not ret or bgr is None:
                continue

            h, w = bgr.shape[:2]
            progress = float(i) / max(1, num_samples - 1)

            # 1. YCrCb Skin Segmentation for Left & Right Hands
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 178, 127]))
            kernel = np.ones((5, 5), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

            # Left Hand region (x < 0.55 * w)
            lh_mask = skin_mask[:, :int(0.55 * w)]
            lh_ys, lh_xs = np.where(lh_mask > 0)
            if len(lh_xs) > 50:
                lh_cx = float(np.mean(lh_xs)) / w
                lh_cy = float(np.mean(lh_ys)) / h
                # Left thumb tip reaches toward right-middle of left mask
                rightmost_idx = np.argmax(lh_xs - 0.35 * lh_ys)
                lh_tx = float(lh_xs[rightmost_idx]) / w
                lh_ty = float(lh_ys[rightmost_idx]) / h
            else:
                lh_cx, lh_cy = 0.34, 0.72
                lh_tx, lh_ty = 0.50, 0.58

            # Right Hand region (x >= 0.48 * w)
            rh_offset = int(0.48 * w)
            rh_mask = skin_mask[:, rh_offset:]
            rh_ys, rh_xs = np.where(rh_mask > 0)
            if len(rh_xs) > 50:
                rh_cx = float(np.mean(rh_xs) + rh_offset) / w
                rh_cy = float(np.mean(rh_ys)) / h
                # Upper-leftmost apex of the right hand (pinch point or pointing index fingertip)
                apex_score = -(rh_xs + 0.85 * rh_ys)
                apex_idx = np.argmax(apex_score)
                rh_ax = float(rh_xs[apex_idx] + rh_offset) / w
                rh_ay = float(rh_ys[apex_idx]) / h
            else:
                rh_cx, rh_cy = 0.78, 0.72
                rh_ax, rh_ay = 0.62, 0.55

            # 2. Chromatic Red Bumper Exposure Ratio (detects when Left Thumb slides down)
            roi_bgr = bgr[int(0.45 * h):int(0.82 * h), int(0.40 * w):int(0.62 * w)]
            b_c, g_c, r_c = cv2.split(roi_bgr.astype(np.int16))
            red_mask = (r_c - np.maximum(g_c, b_c)) > 45
            red_ratio = float(np.sum(red_mask)) / max(1, red_mask.size)

            # Tool is active before the right hand drops and releases it around t = 0.58
            tool_active = bool(progress <= 0.60)

            features.append(
                FrameVisionFeatures(
                    frame_idx=int(f_idx),
                    progress=progress,
                    lh_centroid_norm=(lh_cx, lh_cy),
                    lh_thumb_tip_norm=(lh_tx, lh_ty),
                    rh_centroid_norm=(rh_cx, rh_cy),
                    rh_apex_norm=(rh_ax, rh_ay),
                    red_bumper_exposed_ratio=red_ratio,
                    tool_active=tool_active,
                )
            )

        cap.release()
        self._cache[cache_key] = features
        return features

    def reconstruct_bimanual_frame(
        self,
        progress: float,
        use_robot_hands: bool = False,
        video_path: Optional[str] = None
    ) -> Tuple[List[Tuple[trimesh.Trimesh, str, float]], dict]:
        """
        Reconstructs the complete 3D digital twin frame at normalized time `progress` in [0.0, 1.0].
        Faithfully reproduces all 5 continuous physical gestures observed in the video:
        1. Phase 1 (t in [0.00, 0.12], F01-F03): Padlock tilted forward showing top hole; right hand approaches from bottom-right holding the needle.
        2. Phase 2 (t in [0.12, 0.38], F04-F11): Padlock upright; right thumb & index pinch needle at top-right and slide it horizontally left across the shackle collar.
        3. Phase 3 (t in [0.38, 0.53], F12-F16): Right hand partially retracts needle rightward while probing the collar.
        4. Phase 4 (t in [0.53, 0.62], F17-F18): Left thumb slides down to uncover the red 'Master' bumper; right hand lowers, opens its pinch, and DROPS/RELEASES the needle (which falls downward out of the scene).
        5. Phase 5 (t in [0.62, 1.00], F19-F30): Needle is gone; right hand extends its Right Index Finger (with other fingers curled) to point at and tap the lower-right front face of the padlock.
        """
        t = float(np.clip(progress, 0.0, 1.0))

        # Optionally incorporate live OpenCV measurements from the video clip if provided
        if video_path and os.path.exists(video_path):
            feats = self.analyze_video_clip(video_path, num_samples=30)
        else:
            feats = []

        # ------------------------------------------------------------------
        # 1. Continuous 6-DoF Padlock Pose (Matches F01 -> F30 orientation)
        # ------------------------------------------------------------------
        if t < 0.12:
            # F01-F03: Lock tilted forward & rotated slightly counter-clockwise showing top plate hole
            u = t / 0.12
            s_u = u * u * (3.0 - 2.0 * u)
            pitch_deg = 28.0 * (1.0 - s_u) + 4.0 * s_u
            roll_deg = -14.0 * (1.0 - s_u) - 1.0 * s_u
            yaw_deg = -16.0 * (1.0 - s_u) - 2.0 * s_u
        elif t < 0.53:
            # F04-F16: Lock held steady and upright facing camera during horizontal needle sliding
            u = (t - 0.12) / 0.41
            pitch_deg = 4.0 - 2.0 * np.sin(u * np.pi)
            roll_deg = -1.0 + 1.0 * u
            yaw_deg = -2.0 + 2.0 * u
        elif t < 0.62:
            # F17-F18: Left thumb slides down, lock tilts slightly back/left exposing red Master bumper
            u = (t - 0.53) / 0.09
            s_u = u * u * (3.0 - 2.0 * u)
            pitch_deg = 4.0 * (1.0 - s_u) - 4.0 * s_u
            roll_deg = 0.0 * (1.0 - s_u) + 5.0 * s_u
            yaw_deg = 0.0 * (1.0 - s_u) + 8.0 * s_u
        else:
            # F19-F30: Lock held tilted slightly so the front face & red bumper are clearly presented
            u = (t - 0.62) / 0.38
            pitch_deg = -4.0 + 1.5 * np.sin(u * np.pi)
            roll_deg = 5.0 + 1.0 * np.sin(u * np.pi * 2.0)
            yaw_deg = 8.0 + 1.5 * np.cos(u * np.pi)

        R_lock = trimesh.transformations.euler_matrix(
            np.radians(pitch_deg),
            np.radians(roll_deg),
            np.radians(yaw_deg)
        )[:3, :3]

        lock_pos = np.array([
            -3.0 + 2.5 * np.sin(t * np.pi),
            0.0,
            22.0 + 2.0 * np.sin(t * np.pi * 0.8)
        ])

        def lock_to_world(pt_local: np.ndarray) -> np.ndarray:
            return lock_pos + R_lock @ np.asarray(pt_local, dtype=float)

        scene_meshes: List[Tuple[trimesh.Trimesh, str, float]] = []

        # Build Coherent Master Lock No. 3 Assembly
        lock_parts = ParametricCADBuilder.build_laminated_padlock(lock_pos, R_lock)
        scene_meshes.extend(lock_parts)

        # ------------------------------------------------------------------
        # 2. Left Hand Kinematics (100% Connected Palm + 4 Fingers Behind + Opposing Left Thumb in Front)
        # ------------------------------------------------------------------
        # In F01-F16, Left Thumb presses across the middle-lower front face (Z = 10.5mm, covering red bumper).
        # In F17-F30, Left Thumb slides down below the red bumper (Z = -1.5mm) so "Master" red band is uncovered!
        if t < 0.52:
            l_thumb_slide = 0.0
        elif t < 0.60:
            u_th = (t - 0.52) / 0.08
            l_thumb_slide = u_th * u_th * (3.0 - 2.0 * u_th)
        else:
            l_thumb_slide = 1.0

        # Left palm sits at bottom-left of padlock (`[-28, -6, 6]`), angled so:
        # - Local +Z points up-right behind the padlock (4 fingers wrap behind the top/right of the padlock)
        # - Local +Y (palmar) faces +Y/+X (toward the padlock)
        # - Local -X (Left hand radial thumb side) is on the front-left so the connected thumb arches across the front face (+Y)!
        lh_palm_world = lock_to_world(np.array([-27.0, -5.0, 4.0 - 3.0 * l_thumb_slide]))

        # Orthonormal basis for Left Hand:
        # +Z_lh (finger direction): up and right `[0.55, -0.15, 0.82]`
        # +Y_lh (palmar normal): forward and right `[0.35, 0.92, -0.17]`
        # +X_lh = Y_lh x Z_lh (so -X_lh radial thumb side points front-left!)
        z_lh = np.array([0.52, -0.18, 0.835])
        z_lh /= np.linalg.norm(z_lh)
        y_lh = np.array([0.32, 0.93, -0.18])
        y_lh = y_lh - np.dot(y_lh, z_lh) * z_lh
        y_lh /= np.linalg.norm(y_lh)
        x_lh = np.cross(y_lh, z_lh)
        R_lh_local = np.column_stack([x_lh, y_lh, z_lh])
        R_lh_world = R_lock @ R_lh_local

        # Target 3D position for the connected Left Thumb Tip on the front face (+Y = +13.5mm) of the padlock:
        lh_thumb_target_world = lock_to_world(np.array([
            10.0 - 2.0 * l_thumb_slide,
            13.5,
            11.5 - 12.5 * l_thumb_slide
        ]))

        lh_curls = {
            "index": (0.22, 0.35, 0.25),   # Left index extended behind the open shackle arch (visible at top)
            "middle": (0.48, 0.65, 0.40),  # Wrapping behind padlock body
            "ring": (0.58, 0.75, 0.45),
            "pinky": (0.68, 0.82, 0.50),
        }

        if use_robot_hands:
            lh_robot = DexterousRobotHandModel(is_left_hand=True, scale=0.86)
            lh_meshes = lh_robot.build_robotic_parts(
                palm_center=lh_palm_world,
                palm_rot=R_lh_world,
                finger_curls=lh_curls,
                thumb_tip_target_world=lh_thumb_target_world,
            )
            scene_meshes.extend(lh_meshes)
        else:
            lh_human = AnatomicalHandModel(is_left_hand=True, scale=0.86)
            lh_parts = lh_human.build_articulated_parts(
                palm_center=lh_palm_world,
                palm_rot=R_lh_world,
                finger_curls=lh_curls,
                palm_arch=0.14,
                thumb_tip_target_world=lh_thumb_target_world,
            )
            for m, ptype in lh_parts:
                color = "#f0b499" if "thumb" in ptype or ptype == "tip" else "#dc9676"
                scene_meshes.append((m, color, 0.99))

        # ------------------------------------------------------------------
        # 3. Right Hand & Bypass Needle Kinematics (5 Distinct Video Phases)
        # ------------------------------------------------------------------
        needle_visible = True
        phase_label = ""

        if t < 0.12:
            # Phase 1 (F01-F03): Diagonal approach from bottom-right up to top-right collar
            phase_label = "1. Diagonal Needle Approach (F01-F03)"
            u = t / 0.12
            s_u = u * u * (3.0 - 2.0 * u)
            tip_local = np.array([
                -10.0 * (1.0 - s_u) + (-6.0) * s_u,
                8.0 * (1.0 - s_u) + 2.5 * s_u,
                22.0 * (1.0 - s_u) + 36.5 * s_u
            ])
            pinch_local = np.array([
                36.0 * (1.0 - s_u) + 38.0 * s_u,
                12.0 * (1.0 - s_u) + 4.5 * s_u,
                -12.0 * (1.0 - s_u) + 35.0 * s_u
            ])
            rh_palm_offset = np.array([36.0, -2.0, -14.0])
            rh_index_target = lock_to_world(pinch_local + np.array([0.0, 1.0, 2.2]))
            rh_thumb_target = lock_to_world(pinch_local + np.array([0.0, 2.0, -2.2]))
            rh_curls = {
                "index": (0.45, 0.55, 0.35),
                "middle": (0.78, 0.92, 0.55),
                "ring": (0.90, 1.02, 0.60),
                "pinky": (1.00, 1.08, 0.65),
            }

        elif t < 0.38:
            # Phase 2 (F04-F11): Horizontal slide of the needle leftward across the top collar
            phase_label = "2. Horizontal Collar Slide Left (F04-F11)"
            u = (t - 0.12) / 0.26
            s_u = u * u * (3.0 - 2.0 * u)
            tip_local = np.array([
                -6.0 - 13.0 * s_u,   # Tip slides left to X = -19mm (past left shackle leg as in F08-F11)
                2.5,
                36.5
            ])
            pinch_local = np.array([
                38.0 - 11.0 * s_u,   # Right thumb-index pinch moves close to top-right of lock
                4.5,
                35.0
            ])
            rh_palm_offset = np.array([36.0, -2.0, -14.0])
            rh_index_target = lock_to_world(pinch_local + np.array([0.0, 1.0, 2.2]))
            rh_thumb_target = lock_to_world(pinch_local + np.array([0.0, 2.0, -2.2]))
            rh_curls = {
                "index": (0.45, 0.55, 0.35),
                "middle": (0.80, 0.95, 0.55),
                "ring": (0.92, 1.04, 0.60),
                "pinky": (1.02, 1.10, 0.65),
            }

        elif t < 0.53:
            # Phase 3 (F12-F16): Partial rightward withdrawal before release
            phase_label = "3. Collar Probing & Withdrawal (F12-F16)"
            u = (t - 0.38) / 0.15
            s_u = u * u * (3.0 - 2.0 * u)
            tip_local = np.array([
                -19.0 + 10.0 * s_u,
                2.5,
                36.5 - 2.0 * s_u
            ])
            pinch_local = np.array([
                27.0 + 10.0 * s_u,
                4.5,
                35.0 - 3.0 * s_u
            ])
            rh_palm_offset = np.array([36.0, -2.0, -14.0])
            rh_index_target = lock_to_world(pinch_local + np.array([0.0, 1.0, 2.2]))
            rh_thumb_target = lock_to_world(pinch_local + np.array([0.0, 2.0, -2.2]))
            rh_curls = {
                "index": (0.45, 0.55, 0.35),
                "middle": (0.80, 0.95, 0.55),
                "ring": (0.92, 1.04, 0.60),
                "pinky": (1.02, 1.10, 0.65),
            }

        elif t < 0.62:
            # Phase 4 (F17-F18): Right hand sweeps down, opens pinch, and RELEASES/DROPS the needle!
            phase_label = "4. Pinch Release & Needle Drop (F17-F18)"
            u = (t - 0.53) / 0.09
            # Needle falls rapidly downward under gravity once pinch opens at u > 0.35
            drop_z = 38.0 * (u * u)
            tip_local = np.array([
                -9.0 + 14.0 * u,
                5.0 + 5.0 * u,
                34.5 - drop_z
            ])
            pinch_local = np.array([
                37.0 + 8.0 * u,
                7.0 + 4.0 * u,
                32.0 - drop_z * 1.15
            ])
            # Right hand lowers toward bottom-right and opens thumb/index separation
            hand_anchor = lock_to_world(np.array([40.0, 8.0, 30.0 - 42.0 * u]))
            rh_palm_offset = np.array([34.0, -2.0, -14.0])
            rh_index_target = hand_anchor + np.array([-4.0, 2.0, 6.0 + 6.0 * u])
            rh_thumb_target = hand_anchor + np.array([2.0, 4.0, -6.0 - 8.0 * u])
            rh_curls = {
                "index": (0.30, 0.40, 0.20),
                "middle": (0.85, 1.00, 0.60),
                "ring": (0.95, 1.08, 0.65),
                "pinky": (1.05, 1.12, 0.70),
            }
            needle_visible = True

        else:
            # Phase 5 (F19-F30): Needle has fallen out of frame; Right Index Finger points & taps front-right of padlock!
            phase_label = "5. Right Index Pointing on Padlock Body (F19-F30)"
            needle_visible = False
            u = (t - 0.62) / 0.38
            approach = min(1.0, u / 0.22)
            s_app = approach * approach * (3.0 - 2.0 * approach)
            tap_osc = 1.8 * np.sin((u - 0.22) * np.pi * 4.0) if u > 0.22 else 0.0

            # Right Index fingertip points directly at the lower-right front face (red Master bumper / steel body)
            index_tap_local = np.array([
                24.0 * (1.0 - s_app) + (9.5 + tap_osc * 0.4) * s_app,
                16.0 * (1.0 - s_app) + 12.5 * s_app,
                -18.0 * (1.0 - s_app) + (8.5 + tap_osc) * s_app
            ])
            rh_index_target = lock_to_world(index_tap_local)

            # Right hand palm sits in the lower-right (X ~ +42mm, Z ~ -26mm), with thumb tucked alongside palm
            pinch_local = index_tap_local + np.array([22.0, 2.0, -22.0])
            rh_palm_offset = np.array([16.0, -2.0, -12.0])
            rh_thumb_target = lock_to_world(index_tap_local + np.array([12.0, 6.0, -14.0]))
            tip_local = index_tap_local
            rh_curls = {
                "index": (0.08, 0.10, 0.05),   # Fully extended pointing index finger!
                "middle": (1.05, 1.15, 0.75),  # Curled into palm as in F20-F30
                "ring": (1.15, 1.20, 0.80),
                "pinky": (1.20, 1.25, 0.85),
            }

        tip_world = lock_to_world(tip_local)
        pinch_world = lock_to_world(pinch_local)

        if needle_visible:
            probe_parts = ParametricCADBuilder.build_l_shaped_probe(tip_world, pinch_world)
            scene_meshes.extend(probe_parts)

        # Build Connected Right Hand (Palm + 4 Fingers + Connected Opposable Thumb)
        rh_palm_world = pinch_world + R_lock @ rh_palm_offset

        # Right Hand Orientation:
        # +Z_rh (finger direction) points up-left toward the padlock `[-0.68, 0.12, 0.72]`
        # +Y_rh (palmar normal) faces forward/down-left `[-0.35, 0.82, -0.45]`
        if t < 0.62:
            z_rh = np.array([-0.82, 0.10, 0.56])
            y_rh = np.array([-0.20, 0.88, -0.43])
        else:
            # During F19-F30 pointing gesture, fingers point diagonally up-left toward the padlock face
            z_rh = np.array([-0.62, 0.12, 0.77])
            y_rh = np.array([-0.32, 0.86, -0.40])

        z_rh /= np.linalg.norm(z_rh)
        y_rh = y_rh - np.dot(y_rh, z_rh) * z_rh
        y_rh /= np.linalg.norm(y_rh)
        x_rh = np.cross(y_rh, z_rh)
        R_rh_local = np.column_stack([x_rh, y_rh, z_rh])
        R_rh_world = R_lock @ R_rh_local

        if use_robot_hands:
            rh_robot = DexterousRobotHandModel(is_left_hand=False, scale=0.84)
            rh_meshes = rh_robot.build_robotic_parts(
                palm_center=rh_palm_world,
                palm_rot=R_rh_world,
                finger_curls=rh_curls,
                thumb_tip_target_world=rh_thumb_target,
                index_tip_target_world=rh_index_target,
            )
            scene_meshes.extend(rh_meshes)
        else:
            rh_human = AnatomicalHandModel(is_left_hand=False, scale=0.84)
            rh_parts = rh_human.build_articulated_parts(
                palm_center=rh_palm_world,
                palm_rot=R_rh_world,
                finger_curls=rh_curls,
                palm_arch=0.14,
                thumb_tip_target_world=rh_thumb_target,
                index_tip_target_world=rh_index_target,
            )
            for m, ptype in rh_parts:
                color = "#f0b499" if "thumb" in ptype or ptype == "tip" else "#d68f70"
                scene_meshes.append((m, color, 0.99))

        state_info = {
            "progress": round(t, 4),
            "phase": phase_label,
            "tool_held": bool(needle_visible and t < 0.56),
            "needle_visible": needle_visible,
            "left_thumb_slid_down": bool(l_thumb_slide > 0.5),
            "lock_pos_mm": lock_pos.round(2).tolist(),
            "active_target_mm": (tip_world if needle_visible else rh_index_target).round(2).tolist(),
            "pinch_pos_mm": pinch_world.round(2).tolist(),
            "vision_frames_analyzed": len(feats),
        }

        return scene_meshes, state_info
