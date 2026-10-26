"""
Anatomical 3D Human Hand & Humanoid Robot Dexterous Hand Models for Praxis.
Constructs natural, physiologically proportioned human hands and authentic humanoid robot hands:
- Anatomical human hand: smooth solid metacarpal palm with palmar arch, organic thenar eminence,
  5 articulated multi-phalanx fingers, and a true opposable thumb with saddle-joint kinematics.
  Strictly eliminates artificial cross-struts and floating segments.
- Dexterous humanoid robot hands (Unitree H1 / Tesla Optimus style): industrial wrist flanges,
  actuator chassis, mechanical linkage phalanges, rotary hinge joints, and high-friction tactile pads.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh


def create_smooth_capsule(
    p_start: np.ndarray,
    p_end: np.ndarray,
    r_start: float,
    r_end: float,
    sections: int = 16
) -> trimesh.Trimesh:
    """
    Creates a smooth tapered capsule connecting p_start to p_end with spherical end caps.
    """
    p_start = np.asarray(p_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    vec = p_end - p_start
    length = float(np.linalg.norm(vec))
    if length < 1e-4:
        s = trimesh.creation.icosphere(subdivisions=1, radius=max(r_start, 0.5))
        s.apply_translation(p_start)
        return s

    r_mid = 0.5 * (r_start + r_end)
    cyl = trimesh.creation.cylinder(radius=r_mid, height=length, sections=sections)
    verts = cyl.vertices.copy()
    t = (verts[:, 2] + length / 2.0) / length
    scale_xy = (r_start * (1.0 - t) + r_end * t) / max(r_mid, 1e-5)
    verts[:, 0] *= scale_xy
    verts[:, 1] *= scale_xy
    cyl.vertices = verts

    s1 = trimesh.creation.icosphere(subdivisions=1, radius=r_start)
    s1.apply_translation([0.0, 0.0, -length / 2.0])
    s2 = trimesh.creation.icosphere(subdivisions=1, radius=r_end)
    s2.apply_translation([0.0, 0.0, length / 2.0])

    seg = trimesh.util.concatenate([cyl, s1, s2])

    z_axis = np.array([0.0, 0.0, 1.0])
    direction = vec / length
    rot_mat = trimesh.geometry.align_vectors(z_axis, direction)
    seg.apply_transform(rot_mat)
    seg.apply_translation(0.5 * (p_start + p_end))
    return seg


create_smooth_segment = create_smooth_capsule


def solve_two_bone_ik(
    p_root: np.ndarray,
    p_target: np.ndarray,
    l1: float,
    l2: float,
    bend_normal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical 2-bone Inverse Kinematics solver in 3D.
    """
    p_root = np.asarray(p_root, dtype=float)
    p_target = np.asarray(p_target, dtype=float)
    d_vec = p_target - p_root
    dist = float(np.linalg.norm(d_vec))
    max_reach = (l1 + l2) * 0.995
    min_reach = abs(l1 - l2) * 1.05

    if dist < 1e-5:
        forward = np.array([0.0, 0.0, 1.0])
        dist = min_reach
    else:
        forward = d_vec / dist

    clamped_dist = float(np.clip(dist, min_reach, max_reach))
    p_end = p_root + forward * clamped_dist

    cos_alpha = (l1 * l1 + clamped_dist * clamped_dist - l2 * l2) / (2.0 * l1 * clamped_dist)
    cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
    sin_alpha = float(np.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha)))

    bn = np.asarray(bend_normal, dtype=float)
    bn_perp = bn - np.dot(bn, forward) * forward
    bn_norm = float(np.linalg.norm(bn_perp))
    if bn_norm < 1e-5:
        fallback = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        bn_perp = fallback - np.dot(fallback, forward) * forward
        bn_norm = float(np.linalg.norm(bn_perp))
    bend_dir = bn_perp / max(bn_norm, 1e-5)

    p_mid = p_root + forward * (l1 * cos_alpha) + bend_dir * (l1 * sin_alpha)
    return p_mid, p_end


class AnthropometricHandBounds:
    """
    Standard human anthropometric constraints and physiological joint limits.
    Based on standard anthropometric surveys (e.g. NASA-STD-3000 / ANSUR II).
    Adult hand length (wrist crease to distal tip of middle finger): 150mm - 250mm.
    """
    MIN_HAND_LENGTH_MM = 150.0
    MAX_HAND_LENGTH_MM = 250.0
    MIN_PALM_BREADTH_MM = 65.0
    MAX_PALM_BREADTH_MM = 95.0
    NOMINAL_HAND_LENGTH_MM = 184.0
    NOMINAL_PALM_BREADTH_MM = 82.0

    # Physiological joint angle limits (radians)
    # Flexion (+) / Extension (-)
    MCP_FLEXION_LIMITS = (-0.15, 1.60)   # -8 deg to 92 deg
    PIP_FLEXION_LIMITS = (-0.05, 1.85)   # -3 deg to 106 deg
    DIP_FLEXION_LIMITS = (-0.05, 1.55)   # -3 deg to 89 deg
    FINGER_SPREAD_LIMITS = (-0.26, 0.26) # +/- 15 deg abduction/adduction

    @classmethod
    def clamp_curl_angles(cls, curl: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Clamps (MCP, PIP, DIP) curl angles to physiological ranges (no backwards bending)."""
        mcp = float(np.clip(curl[0], cls.MCP_FLEXION_LIMITS[0], cls.MCP_FLEXION_LIMITS[1]))
        pip = float(np.clip(curl[1], cls.PIP_FLEXION_LIMITS[0], cls.PIP_FLEXION_LIMITS[1]))
        dip = float(np.clip(curl[2], cls.DIP_FLEXION_LIMITS[0], cls.DIP_FLEXION_LIMITS[1]))
        return (mcp, pip, dip)

    @classmethod
    def clamp_spread(cls, spread: float) -> float:
        """Clamps finger abduction/adduction angle."""
        return float(np.clip(spread, cls.FINGER_SPREAD_LIMITS[0], cls.FINGER_SPREAD_LIMITS[1]))


class AnatomicalHandModel:
    """
    Continuous, anatomically proportioned human hand with a solid metacarpal palm,
    muscular thenar eminence, 4 articulated fingers, and an opposable thumb.
    Standardized to 50th percentile male hand metrics (scaled mm).
    """

    def __init__(self, is_left_hand: bool = False, scale: float = 1.0, enforce_anthropometrics: bool = True):
        self.is_left = is_left_hand
        self.scale = scale
        
        # Validate that scaled hand dimensions stay strictly within human adult bounds
        hand_length = self.total_hand_length_mm
        if enforce_anthropometrics:
            if hand_length < AnthropometricHandBounds.MIN_HAND_LENGTH_MM or hand_length > AnthropometricHandBounds.MAX_HAND_LENGTH_MM:
                raise ValueError(
                    f"Unphysical hand scale {scale:.2f} results in hand length {hand_length:.1f}mm, "
                    f"outside human adult bounds [{AnthropometricHandBounds.MIN_HAND_LENGTH_MM}mm, {AnthropometricHandBounds.MAX_HAND_LENGTH_MM}mm]."
                )

    @property
    def total_hand_length_mm(self) -> float:
        """Total hand length from wrist crease to tip of middle finger in millimeters."""
        return AnthropometricHandBounds.NOMINAL_HAND_LENGTH_MM * self.scale

    @property
    def palm_breadth_mm(self) -> float:
        """Palm breadth across MCP knuckles in millimeters."""
        return AnthropometricHandBounds.NOMINAL_PALM_BREADTH_MM * self.scale

    def build_articulated_parts(
        self,
        palm_center: np.ndarray,
        palm_rot: np.ndarray,
        finger_curls: Optional[Dict[str, Tuple[float, float, float]]] = None,
        thumb_angles: Optional[Tuple[float, float, float]] = None,
        thumb_tip_target_world: Optional[np.ndarray] = None,
        index_tip_target_world: Optional[np.ndarray] = None,
        finger_spread: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[trimesh.Trimesh, str]]:
        """
        Builds natural hand meshes in world coordinates.
        Local frame convention:
        - Origin: Center of palm metacarpals
        - +Z: Distal (towards fingers, knuckles at Z ~ +25mm)
        - -Z: Proximal (towards wrist, wrist crease at Z ~ -25mm)
        - +Y: Palmar (inner palm surface facing object)
        - -Y: Dorsal (back of hand)
        - +X (for right hand) / -X (for left hand): Radial (thumb side)
        """
        s = self.scale
        sign = -1.0 if self.is_left else 1.0
        parts: List[Tuple[trimesh.Trimesh, str]] = []

        palm_center = np.asarray(palm_center, dtype=float)
        palm_rot = np.asarray(palm_rot, dtype=float)
        R_inv = palm_rot.T

        def world_to_local(pt_w: np.ndarray) -> np.ndarray:
            return R_inv @ (np.asarray(pt_w, dtype=float) - palm_center)

        if finger_curls is None:
            raw_curls = {
                "index": (0.60, 0.75, 0.45),
                "middle": (0.70, 0.85, 0.50),
                "ring": (0.80, 0.95, 0.55),
                "pinky": (0.90, 1.00, 0.60),
            }
        else:
            raw_curls = finger_curls

        clamped_curls = {
            fname: AnthropometricHandBounds.clamp_curl_angles(raw_curls.get(fname, (0.6, 0.8, 0.5)))
            for fname in ["index", "middle", "ring", "pinky"]
        }

        if finger_spread is None:
            raw_spread = {"index": 0.05, "middle": 0.0, "ring": -0.05, "pinky": -0.10}
        else:
            raw_spread = finger_spread

        clamped_spread = {
            fname: AnthropometricHandBounds.clamp_spread(raw_spread.get(fname, 0.0))
            for fname in ["index", "middle", "ring", "pinky"]
        }

        # -------------------------------------------------------------
        # 1. Metacarpal Palm (Smooth cylindrical anatomical rays + carpal arch)
        # -------------------------------------------------------------
        # Carpal wrist crease anchor points
        p_wrist_mid = np.array([0.0, -2.0 * s, -22.0 * s])
        p_forearm = np.array([-2.0 * sign * s, -3.0 * s, -55.0 * s])
        parts.append((create_smooth_capsule(p_wrist_mid, p_forearm, 15.0 * s, 17.5 * s, sections=16), "palm"))

        # Knuckle joint positions along the distal edge of the metacarpals (Z = +24mm)
        knuckle_x = {
            "index": 15.5 * sign * s,
            "middle": 5.0 * sign * s,
            "ring": -5.5 * sign * s,
            "pinky": -15.0 * sign * s,
        }
        knuckle_z = {
            "index": 22.0 * s,
            "middle": 24.5 * s,
            "ring": 22.5 * s,
            "pinky": 19.5 * s,
        }

        mcp_points: Dict[str, np.ndarray] = {}
        # Construct palm from 4 anatomical metacarpal bone rays with smooth condyles
        for fname in ["index", "middle", "ring", "pinky"]:
            kx = knuckle_x[fname]
            cup_y = (abs(kx) / 20.0) * 1.5 * s
            p_mcp = np.array([kx, cup_y, knuckle_z[fname]])
            mcp_points[fname] = p_mcp
            
            # Metacarpal shaft from wrist to knuckle (nested cylinder)
            p_carpal_origin = np.array([kx * 0.65, -1.5 * s, -20.0 * s])
            parts.append((create_smooth_capsule(p_carpal_origin, p_mcp, 7.5 * s, 6.5 * s, sections=14), "palm"))
            # Spherical condyle joint cap at MCP
            parts.append((create_smooth_capsule(p_mcp, p_mcp, 6.2 * s, 6.2 * s, sections=12), "palm"))

        # Smooth central palmar pad filler
        p_palm_center_1 = np.array([0.0, -1.0 * s, -2.0 * s])
        p_palm_center_2 = np.array([0.0, 0.5 * s, 14.0 * s])
        parts.append((create_smooth_capsule(p_palm_center_1, p_palm_center_2, 12.0 * s, 10.5 * s, sections=16), "palm"))

        # -------------------------------------------------------------
        # 2. Four Articulated Fingers (Index, Middle, Ring, Pinky)
        # -------------------------------------------------------------
        finger_dims = {
            "index": (24.0 * s, 16.0 * s, 12.0 * s, 5.8 * s),
            "middle": (26.5 * s, 17.5 * s, 13.0 * s, 6.0 * s),
            "ring": (24.5 * s, 16.5 * s, 12.0 * s, 5.7 * s),
            "pinky": (19.5 * s, 13.0 * s, 10.0 * s, 5.0 * s),
        }

        p_pip_dict: Dict[str, np.ndarray] = {}
        p_dip_dict: Dict[str, np.ndarray] = {}
        p_tip_dict: Dict[str, np.ndarray] = {}

        for fname, (l_prox, l_mid, l_dist, r_base) in finger_dims.items():
            p_mcp = mcp_points[fname]

            if fname == "index" and index_tip_target_world is not None:
                # Analytical IK for index pointing / contact
                target_local = world_to_local(index_tip_target_world)
                vec_tot = target_local - p_mcp
                dist_tot = float(np.linalg.norm(vec_tot))
                dir_tot = vec_tot / max(dist_tot, 1e-5)
                dip_dir = dir_tot + np.array([0.0, 0.20, 0.0])
                dip_dir /= max(float(np.linalg.norm(dip_dir)), 1e-5)
                p_dip_target = target_local - dip_dir * l_dist
                p_pip, p_dip = solve_two_bone_ik(
                    p_mcp, p_dip_target, l_prox, l_mid, bend_normal=np.array([0.0, -0.7, 0.7])
                )
                p_tip = target_local if dist_tot <= (l_prox + l_mid + l_dist) else (p_dip + dip_dir * l_dist)
            else:
                mcp_a, pip_a, dip_a = clamped_curls[fname]
                sp = clamped_spread[fname] * sign
                dir1 = np.array([np.sin(sp), np.sin(mcp_a), np.cos(mcp_a) * np.cos(sp)])
                dir1 /= np.linalg.norm(dir1)
                p_pip = p_mcp + dir1 * l_prox

                a2 = mcp_a + pip_a
                dir2 = np.array([np.sin(sp) * 0.5, np.sin(a2), np.cos(a2)])
                dir2 /= np.linalg.norm(dir2)
                p_dip = p_pip + dir2 * l_mid

                a3 = a2 + dip_a
                dir3 = np.array([0.0, np.sin(a3), np.cos(a3)])
                dir3 /= np.linalg.norm(dir3)
                p_tip = p_dip + dir3 * l_dist

            p_pip_dict[fname] = p_pip
            p_dip_dict[fname] = p_dip
            p_tip_dict[fname] = p_tip

            parts.append((create_smooth_capsule(p_mcp, p_pip, r_base, r_base * 0.90, sections=14), "phalanx"))
            parts.append((create_smooth_capsule(p_pip, p_dip, r_base * 0.90, r_base * 0.80, sections=14), "phalanx"))
            parts.append((create_smooth_capsule(p_dip, p_tip, r_base * 0.80, r_base * 0.68, sections=14), "tip"))

        # -------------------------------------------------------------
        # 3. Opposable Thumb (Organic Muscular Thenar + 2 Phalanges)
        # -------------------------------------------------------------
        # CMC joint sits on the radial-palmar side of the wrist, safely outside workpiece boundary
        p_cmc = np.array([22.0 * sign * s, 2.0 * s, -14.0 * s])
        # Thenar muscle eminence: smooth fleshy muscle connecting wrist to thumb MCP
        p_thenar_mcp = np.array([29.0 * sign * s, 9.0 * s, 4.0 * s])
        thenar_mesh = create_smooth_capsule(p_cmc, p_thenar_mcp, 9.5 * s, 7.5 * s, sections=14)
        parts.append((thenar_mesh, "thumb_base"))

        l_th_prox = 22.0 * s
        l_th_dist = 18.0 * s

        if thumb_tip_target_world is not None:
            # Analytical IK for thumb contact pad reaching target without artificial struts
            th_target_local = world_to_local(thumb_tip_target_world)
            vec_to_t = th_target_local - p_cmc
            dist_to_t = float(np.linalg.norm(vec_to_t))
            dir_to_t = vec_to_t / max(dist_to_t, 1e-5)
            # Route thenar cleanly outside workpiece boundary around the lateral flank
            thenar_dir = 0.50 * dir_to_t + 0.50 * np.array([1.2 * sign, 0.7, 0.2])
            thenar_dir /= max(float(np.linalg.norm(thenar_dir)), 1e-5)
            p_thenar_mcp = p_cmc + thenar_dir * (24.0 * s)

            p_th_ip, p_th_tip = solve_two_bone_ik(
                p_thenar_mcp,
                th_target_local,
                l_th_prox,
                l_th_dist,
                bend_normal=np.array([1.4 * sign, 0.8, 0.1])
            )
        else:
            if thumb_angles is None:
                thumb_angles = (0.60, 0.40, 0.30)
            th_abd, th_mcp, th_ip = thumb_angles
            th_dir1 = np.array([
                sign * np.cos(th_abd) * 0.4,
                np.sin(th_abd),
                np.cos(th_mcp)
            ])
            th_dir1 /= np.linalg.norm(th_dir1)
            p_th_ip = p_thenar_mcp + th_dir1 * l_th_prox

            th_dir2 = np.array([
                -sign * np.sin(th_ip) * 0.4,
                np.sin(th_abd - th_ip * 0.4),
                np.cos(th_mcp + th_ip * 0.3)
            ])
            th_dir2 /= np.linalg.norm(th_dir2)
            p_th_tip = p_th_ip + th_dir2 * l_th_dist

        # Thumb proximal phalanx and distal phalanx (clean, no struts to index)
        parts.append((create_smooth_capsule(p_thenar_mcp, p_th_ip, 7.8 * s, 7.0 * s, sections=14), "thumb"))
        parts.append((create_smooth_capsule(p_th_ip, p_th_tip, 7.0 * s, 6.0 * s, sections=14), "thumb_tip"))

        # Transform all sub-meshes into world frame
        T = np.eye(4)
        T[:3, :3] = palm_rot
        T[:3, 3] = palm_center

        transformed_parts: List[Tuple[trimesh.Trimesh, str]] = []
        for mesh, ptype in parts:
            m_copy = mesh.copy()
            m_copy.apply_transform(T)
            transformed_parts.append((m_copy, ptype))

        # 21 Anatomical Skeletal Joints in world coordinates
        joint_points_local = [
            p_wrist_mid,                          # 0: wrist
            mcp_points["index"],                  # 1: mcp_index
            p_pip_dict["index"],                  # 2: pip_index
            p_dip_dict["index"],                  # 3: dip_index
            p_tip_dict["index"],                  # 4: tip_index
            mcp_points["middle"],                 # 5: mcp_middle
            p_pip_dict["middle"],                 # 6: pip_middle
            p_dip_dict["middle"],                 # 7: dip_middle
            p_tip_dict["middle"],                 # 8: tip_middle
            mcp_points["ring"],                   # 9: mcp_ring
            p_pip_dict["ring"],                   # 10: pip_ring
            p_dip_dict["ring"],                   # 11: dip_ring
            p_tip_dict["ring"],                   # 12: tip_ring
            mcp_points["pinky"],                  # 13: mcp_pinky
            p_pip_dict["pinky"],                  # 14: pip_pinky
            p_dip_dict["pinky"],                  # 15: dip_pinky
            p_tip_dict["pinky"],                  # 16: tip_pinky
            p_cmc,                                # 17: cmc_thumb
            p_thenar_mcp,                         # 18: mcp_thumb
            p_th_ip,                              # 19: ip_thumb
            p_th_tip,                             # 20: tip_thumb
        ]
        self.last_skeleton_joints = [
            (palm_center + palm_rot @ pt).round(2).tolist() for pt in joint_points_local
        ]

        return transformed_parts

    def get_skeleton_joints(self) -> List[List[float]]:
        """Returns the 21 physiological joints in world coordinates."""
        return getattr(self, "last_skeleton_joints", [])


class DexterousRobotHandModel:
    """
    Authentic Humanoid Robot Manipulator (Unitree H1 / Tesla Optimus embodiment).
    Models industrial metallic arm links, rotary wrist actuators, structural palm chassis,
    and mechanical articulated fingers with rotary joint pins and tactile sensor pads.
    """

    def __init__(self, is_left_hand: bool = False, scale: float = 1.0):
        self.is_left = is_left_hand
        self.scale = scale

    def build_robotic_manipulator(
        self,
        palm_center: np.ndarray,
        palm_rot: np.ndarray,
        finger_curls: Optional[Dict[str, Tuple[float, float, float]]] = None,
        thumb_tip_target_world: Optional[np.ndarray] = None,
        index_tip_target_world: Optional[np.ndarray] = None,
    ) -> List[Tuple[trimesh.Trimesh, str, float]]:
        """
        Synthesizes a realistic humanoid robot arm end-effector and dexterous hand
        with distinct mechanical joints and actuators (Unitree H1 / Optimus architecture).
        Returns list of (mesh, hex_color, alpha).
        """
        s = self.scale
        sign = -1.0 if self.is_left else 1.0
        parts: List[Tuple[trimesh.Trimesh, str, float]] = []

        palm_center = np.asarray(palm_center, dtype=float)
        palm_rot = np.asarray(palm_rot, dtype=float)
        R_inv = palm_rot.T

        def world_to_local(pt_w: np.ndarray) -> np.ndarray:
            return R_inv @ (np.asarray(pt_w, dtype=float) - palm_center)

        # 1. Robotic Arm Forearm Link & Wrist Flange (Carbon fiber / dark titanium)
        # Forearm housing: length 70mm, radius 22mm
        forearm_p0 = np.array([0.0, 0.0, -25.0 * s])
        forearm_p1 = np.array([-4.0 * sign * s, -4.0 * s, -85.0 * s])
        forearm_tube = create_smooth_capsule(forearm_p0, forearm_p1, 20.0 * s, 22.0 * s, sections=16)
        parts.append((forearm_tube, "#212121", 0.99))

        # Gold / bronze rotary wrist pitch/yaw actuator ring
        wrist_ring = trimesh.creation.cylinder(radius=21.0 * s, height=8.0 * s, sections=20)
        wrist_ring.apply_translation([0.0, 0.0, -25.0 * s])
        parts.append((wrist_ring, "#d4af37", 0.99))

        # 2. Robotic Palm Chassis (Industrial machined anodized aluminum)
        palm_block = trimesh.creation.box(extents=[44.0 * s, 18.0 * s, 42.0 * s])
        palm_block.apply_translation([0.0, -1.0 * s, 0.0])
        parts.append((palm_block, "#37474f", 0.98))

        # Metallic palm faceplate and servomotor accent ribs
        face_plate = trimesh.creation.box(extents=[38.0 * s, 2.0 * s, 36.0 * s])
        face_plate.apply_translation([0.0, -10.0 * s, 0.0])
        parts.append((face_plate, "#cfd8dc", 0.99))

        # 3. Articulated Robotic Fingers with Bronze Hinge Pins
        knuckle_x = {
            "index": 16.0 * sign * s,
            "middle": 5.5 * sign * s,
            "ring": -5.5 * sign * s,
            "pinky": -15.5 * sign * s,
        }
        finger_lengths = {
            "index": (22.0 * s, 15.0 * s, 12.0 * s),
            "middle": (25.0 * s, 17.0 * s, 13.0 * s),
            "ring": (23.0 * s, 15.5 * s, 12.0 * s),
            "pinky": (18.0 * s, 12.0 * s, 10.0 * s),
        }

        if finger_curls is None:
            finger_curls = {
                "index": (0.6, 0.8, 0.5),
                "middle": (0.7, 0.9, 0.5),
                "ring": (0.8, 1.0, 0.6),
                "pinky": (0.9, 1.0, 0.6),
            }

        for fname in ["index", "middle", "ring", "pinky"]:
            kx = knuckle_x[fname]
            p_mcp = np.array([kx, 0.0, 21.0 * s])
            l_prox, l_mid, l_dist = finger_lengths[fname]

            # Knuckle hinge bracket (bronze pivot pin)
            pin = trimesh.creation.cylinder(radius=3.5 * s, height=10.0 * s, sections=12)
            R_pin = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
            pin.apply_transform(R_pin)
            pin.apply_translation(p_mcp)
            parts.append((pin, "#d4af37", 0.99))

            if fname == "index" and index_tip_target_world is not None:
                target_local = world_to_local(index_tip_target_world)
                vec_tot = target_local - p_mcp
                dist_tot = float(np.linalg.norm(vec_tot))
                dir_tot = vec_tot / max(dist_tot, 1e-5)
                dip_dir = dir_tot + np.array([0.0, 0.15, 0.0])
                dip_dir /= max(float(np.linalg.norm(dip_dir)), 1e-5)
                p_dip_target = target_local - dip_dir * l_dist
                p_pip, p_dip = solve_two_bone_ik(
                    p_mcp, p_dip_target, l_prox, l_mid, bend_normal=np.array([0.0, -0.7, 0.7])
                )
                p_tip = target_local if dist_tot <= (l_prox + l_mid + l_dist) else (p_dip + dip_dir * l_dist)
            else:
                mcp_a, pip_a, dip_a = finger_curls.get(fname, (0.6, 0.8, 0.5))
                dir1 = np.array([0.0, np.sin(mcp_a), np.cos(mcp_a)])
                p_pip = p_mcp + dir1 * l_prox
                a2 = mcp_a + pip_a
                dir2 = np.array([0.0, np.sin(a2), np.cos(a2)])
                p_dip = p_pip + dir2 * l_mid
                a3 = a2 + dip_a
                dir3 = np.array([0.0, np.sin(a3), np.cos(a3)])
                p_tip = p_dip + dir3 * l_dist

            # Mechanical phalanx links (brushed metallic grey)
            parts.append((create_smooth_capsule(p_mcp, p_pip, 5.5 * s, 5.0 * s, 12), "#78909c", 0.99))
            parts.append((create_smooth_capsule(p_pip, p_dip, 5.0 * s, 4.4 * s, 12), "#90a4ae", 0.99))
            # Tactile sensor pad (cyan high-friction surface)
            parts.append((create_smooth_capsule(p_dip, p_tip, 4.4 * s, 3.8 * s, 12), "#00e5ff", 0.99))

        # 4. Robotic Opposable Thumb with Rotary Base Actuator
        p_th_base = np.array([19.0 * sign * s, 2.0 * s, -10.0 * s])
        # Gold rotary motor housing at thumb base
        motor = trimesh.creation.cylinder(radius=7.5 * s, height=12.0 * s, sections=16)
        motor.apply_translation(p_th_base)
        parts.append((motor, "#ffb300", 0.99))

        p_th_mcp = np.array([27.0 * sign * s, 8.0 * s, 4.0 * s])
        parts.append((create_smooth_capsule(p_th_base, p_th_mcp, 6.5 * s, 5.8 * s, 12), "#546e7a", 0.99))

        l_th1 = 20.0 * s
        l_th2 = 16.0 * s

        if thumb_tip_target_world is not None:
            th_target_local = world_to_local(thumb_tip_target_world)
            vec_to_t = th_target_local - p_th_base
            dir_to_t = vec_to_t / max(float(np.linalg.norm(vec_to_t)), 1e-5)
            p_th_mcp = p_th_base + dir_to_t * (18.0 * s)
            p_th_ip, p_th_tip = solve_two_bone_ik(
                p_th_mcp, th_target_local, l_th1, l_th2, bend_normal=np.array([0.5 * sign, 0.6, -0.3])
            )
        else:
            th_dir = np.array([sign * 0.35, 0.65, 0.65])
            th_dir /= np.linalg.norm(th_dir)
            p_th_ip = p_th_mcp + th_dir * l_th1
            p_th_tip = p_th_ip + th_dir * l_th2

        parts.append((create_smooth_capsule(p_th_mcp, p_th_ip, 5.8 * s, 5.0 * s, 12), "#78909c", 0.99))
        parts.append((create_smooth_capsule(p_th_ip, p_th_tip, 5.0 * s, 4.2 * s, 12), "#00e5ff", 0.99))

        # Transform to world coordinate frame
        T = np.eye(4)
        T[:3, :3] = palm_rot
        T[:3, 3] = palm_center

        transformed_robot: List[Tuple[trimesh.Trimesh, str, float]] = []
        for mesh, color, alpha in parts:
            m_copy = mesh.copy()
            m_copy.apply_transform(T)
            transformed_robot.append((m_copy, color, alpha))

        return transformed_robot
