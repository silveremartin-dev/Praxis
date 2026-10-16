"""
Anatomical 3D Hand Mesh & Kinematic Chain Generator for Praxis.
Constructs realistic, articulated multi-segment human hands:
- Metacarpal palm with slight palmar arch deformation
- Opposable thumb (CMC, MCP, IP joints + distal pad)
- Four articulated fingers (Index, Middle, Ring, Pinky) with MCP, PIP, DIP joints
Supports both unified Trimesh export (for OBJ/SDF) and shaded sub-part lists (for high-fidelity 3D rendering).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh


def create_smooth_segment(p_start: np.ndarray, p_end: np.ndarray, r_start: float, r_end: float, sections: int = 16) -> trimesh.Trimesh:
    """
    Creates a smooth tapered capsule segment connecting 3D point p_start to p_end.
    """
    vec = p_end - p_start
    length = float(np.linalg.norm(vec))
    if length < 1e-4:
        s = trimesh.creation.icosphere(subdivisions=1, radius=r_start)
        s.apply_translation(p_start)
        return s

    # Create cylinder along Z
    r_mid = 0.5 * (r_start + r_end)
    cyl = trimesh.creation.cylinder(radius=r_mid, height=length, sections=sections)
    # Taper cylinder vertices slightly from bottom (-length/2) to top (+length/2)
    verts = cyl.vertices.copy()
    t = (verts[:, 2] + length / 2.0) / length  # 0 at bottom, 1 at top
    scale_xy = (r_start * (1.0 - t) + r_end * t) / r_mid
    verts[:, 0] *= scale_xy
    verts[:, 1] *= scale_xy
    cyl.vertices = verts

    s1 = trimesh.creation.icosphere(subdivisions=1, radius=r_start)
    s1.apply_translation([0, 0, -length / 2.0])
    s2 = trimesh.creation.icosphere(subdivisions=1, radius=r_end)
    s2.apply_translation([0, 0, length / 2.0])

    seg = trimesh.util.concatenate([cyl, s1, s2])

    # Align Z axis with vec
    z_axis = np.array([0.0, 0.0, 1.0])
    direction = vec / length
    rot_mat = trimesh.geometry.align_vectors(z_axis, direction)
    seg.apply_transform(rot_mat)
    seg.apply_translation(0.5 * (p_start + p_end))
    return seg


class AnatomicalHandModel:
    """
    Articulated 3D human hand with 5 fingers (including a true opposable thumb)
    and continuous joint kinematics.
    """
    def __init__(self, is_left_hand: bool = False, scale: float = 1.0):
        self.is_left = is_left_hand
        self.scale = scale

    def build_articulated_parts(
        self,
        palm_center: np.ndarray,
        palm_rot: np.ndarray,
        finger_curls: Optional[Dict[str, Tuple[float, float, float]]] = None,
        thumb_angles: Optional[Tuple[float, float, float]] = None,
        palm_arch: float = 0.12
    ) -> List[Tuple[trimesh.Trimesh, str]]:
        """
        Builds articulated hand sub-meshes with realistic joint continuity.
        Returns a list of (trimesh.Trimesh, part_type) where part_type is 'palm', 'phalanx', 'joint', or 'thumb'.

        Local Hand Coordinate Frame (before palm_rot & palm_center):
        - Origin (0,0,0): Center of palm
        - +Z: Towards fingers (metacarpal-phalangeal knuckles at Z = +28mm)
        - -Z: Towards wrist (wrist base at Z = -28mm)
        - +Y: Palmar direction (inner palm facing object)
        - -Y: Dorsal direction (back of hand)
        - +X (for right hand) / -X (for left hand): Radial side (Thumb side)
        """
        s = self.scale
        sign = -1.0 if self.is_left else 1.0
        parts: List[Tuple[trimesh.Trimesh, str]] = []

        # Default curls (mcp_rad, pip_rad, dip_rad) flexing towards +Y (palmar)
        if finger_curls is None:
            finger_curls = {
                "index": (0.7, 0.9, 0.5),
                "middle": (0.8, 1.0, 0.6),
                "ring": (0.9, 1.1, 0.6),
                "pinky": (1.0, 1.1, 0.6),
            }

        # 1. Deformable Metacarpal Palm (composed of 4 overlapping metacarpal rays + palm core)
        # This models the slight palmar cupping/arch outside of the finger joints.
        knuckle_x = {
            "index": 18.0 * sign * s,
            "middle": 6.0 * sign * s,
            "ring": -6.0 * sign * s,
            "pinky": -17.0 * sign * s,
        }
        wrist_pt = np.array([0.0, -2.0 * s, -28.0 * s])

        # Palm core block with rounded corners via metacarpal capsules
        palm_box = trimesh.creation.box(extents=[40.0 * s, 14.0 * s, 46.0 * s])
        palm_box.apply_translation([0.0, 0.0, 2.0 * s])
        parts.append((palm_box, "palm"))

        # Wrist forearm stub
        forearm_end = np.array([-4.0 * sign * s, -4.0 * s, -62.0 * s])
        wrist_seg = create_smooth_segment(wrist_pt, forearm_end, 18.0 * s, 21.0 * s, sections=16)
        parts.append((wrist_seg, "palm"))

        # 2. Four Fingers (Index, Middle, Ring, Pinky)
        finger_dims = {
            "index":  (24.0 * s, 16.0 * s, 12.0 * s, 6.2 * s),
            "middle": (27.0 * s, 18.0 * s, 12.5 * s, 6.5 * s),
            "ring":   (25.0 * s, 17.0 * s, 12.0 * s, 6.0 * s),
            "pinky":  (20.0 * s, 13.0 * s, 10.5 * s, 5.3 * s),
        }

        for fname, (l_prox, l_mid, l_dist, r_base) in finger_dims.items():
            kx = knuckle_x[fname]
            # Slight palmar cupping on outer fingers (ring, pinky)
            cup_y = abs(kx) * palm_arch
            p_mcp = np.array([kx, cup_y, 26.0 * s])

            # Connect wrist to knuckle to form smooth palm web
            meta_seg = create_smooth_segment(wrist_pt, p_mcp, 11.0 * s, r_base * 1.15, sections=12)
            parts.append((meta_seg, "palm"))

            mcp_a, pip_a, dip_a = finger_curls.get(fname, (0.6, 0.8, 0.5))

            # Proximal phalanx
            dir1 = np.array([0.0, np.sin(mcp_a), np.cos(mcp_a)])
            p_pip = p_mcp + dir1 * l_prox
            seg_prox = create_smooth_segment(p_mcp, p_pip, r_base, r_base * 0.90, sections=14)
            parts.append((seg_prox, "phalanx"))

            # Middle phalanx
            a2 = mcp_a + pip_a
            dir2 = np.array([0.0, np.sin(a2), np.cos(a2)])
            p_dip = p_pip + dir2 * l_mid
            seg_mid = create_smooth_segment(p_pip, p_dip, r_base * 0.90, r_base * 0.80, sections=14)
            parts.append((seg_mid, "phalanx"))

            # Distal phalanx (fingertip)
            a3 = a2 + dip_a
            dir3 = np.array([0.0, np.sin(a3), np.cos(a3)])
            p_tip = p_dip + dir3 * l_dist
            seg_dist = create_smooth_segment(p_dip, p_tip, r_base * 0.80, r_base * 0.68, sections=14)
            parts.append((seg_dist, "tip"))

        # 3. Opposable Thumb (CMC base on radial side of wrist, flexing across/upwards)
        # thumb_angles = (cmc_abduction, mcp_flexion, ip_flexion)
        if thumb_angles is None:
            thumb_angles = (0.65, 0.45, 0.35)
        th_abd, th_mcp, th_ip = thumb_angles

        p_cmc = np.array([20.0 * sign * s, 4.0 * s, -12.0 * s])
        # Thenar eminence (muscular pad at base of thumb)
        p_thenar = np.array([24.0 * sign * s, 10.0 * s, 8.0 * s])
        seg_thenar = create_smooth_segment(p_cmc, p_thenar, 12.0 * s, 9.5 * s, sections=14)
        parts.append((seg_thenar, "thumb_base"))

        # Thumb proximal phalanx
        th_dir1 = np.array([
            sign * np.cos(th_abd) * 0.35,
            np.sin(th_abd),
            np.cos(th_mcp)
        ])
        th_dir1 = th_dir1 / np.linalg.norm(th_dir1)
        p_th_ip = p_thenar + th_dir1 * (22.0 * s)
        seg_th_prox = create_smooth_segment(p_thenar, p_th_ip, 8.2 * s, 7.5 * s, sections=14)
        parts.append((seg_th_prox, "thumb"))

        # Thumb distal phalanx (pad)
        th_dir2 = np.array([
            -sign * np.sin(th_ip) * 0.4,
            np.sin(th_abd - th_ip * 0.5),
            np.cos(th_mcp + th_ip * 0.4)
        ])
        th_dir2 = th_dir2 / np.linalg.norm(th_dir2)
        p_th_tip = p_th_ip + th_dir2 * (19.0 * s)
        seg_th_dist = create_smooth_segment(p_th_ip, p_th_tip, 7.5 * s, 6.5 * s, sections=14)
        parts.append((seg_th_dist, "thumb_tip"))

        # Transform all parts into world coordinates
        T = np.eye(4)
        T[:3, :3] = palm_rot
        T[:3, 3] = palm_center

        transformed_parts = []
        for mesh, ptype in parts:
            m_copy = mesh.copy()
            m_copy.apply_transform(T)
            transformed_parts.append((m_copy, ptype))

        return transformed_parts

    def build_hand_mesh(
        self,
        wrist_pos: np.ndarray,
        wrist_rot_matrix: np.ndarray,
        finger_curl: float = 0.5,
        thumb_oppose: float = 0.6,
        skin_color: Tuple[int, int, int, int] = (235, 175, 155, 255)
    ) -> trimesh.Trimesh:
        """
        Builds a single unified Trimesh of the hand (compatible with OBJ export and SDF checks).
        """
        c = finger_curl
        curls = {
            "index":  (0.5 * c, 0.9 * c, 0.6 * c),
            "middle": (0.6 * c, 1.0 * c, 0.6 * c),
            "ring":   (0.7 * c, 1.1 * c, 0.7 * c),
            "pinky":  (0.8 * c, 1.1 * c, 0.7 * c),
        }
        th_angles = (0.5 + 0.4 * thumb_oppose, 0.4 * thumb_oppose, 0.4 * thumb_oppose)
        parts = self.build_articulated_parts(
            palm_center=np.asarray(wrist_pos, dtype=float),
            palm_rot=np.asarray(wrist_rot_matrix, dtype=float),
            finger_curls=curls,
            thumb_angles=th_angles
        )
        combined = trimesh.util.concatenate([m for m, _ in parts])
        combined.visual.face_colors = list(skin_color)
        return combined
