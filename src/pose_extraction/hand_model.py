"""
Anatomical 3D Human Hand & 5-Finger Dexterous Robot Hand Models for Praxis.
Guarantees 100% topological and kinematic continuity:
- Every finger and the opposable thumb are rooted directly on the palm/wrist skeleton
  with bridging metacarpal/thenar web geometry (zero detached segments).
- Supports analytical 3-segment target IK for the opposable thumb and index finger so
  contact pads reach exact 3D manipulation targets while remaining organically attached to the palm.
- Includes `DexterousRobotHandModel` (5-finger anthropomorphic humanoid robot hand with
  metallic phalanges, rotary joint actuators, and tactile fingertip pads) for end-to-end robot simulation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh


def create_smooth_segment(
    p_start: np.ndarray,
    p_end: np.ndarray,
    r_start: float,
    r_end: float,
    sections: int = 16
) -> trimesh.Trimesh:
    """
    Creates a smooth tapered capsule connecting 3D point p_start to p_end with spherical joint caps.
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


def solve_two_bone_ik(
    p_root: np.ndarray,
    p_target: np.ndarray,
    l1: float,
    l2: float,
    bend_normal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analytical 2-bone Inverse Kinematics solver in 3D.
    Given a root joint `p_root`, a target tip `p_target`, bone lengths `l1`, `l2`,
    and a preferred bending plane normal/direction `bend_normal`, returns `(p_mid, p_end)`
    such that `||p_mid - p_root|| == l1` and `||p_end - p_mid|| == l2`.
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

    # Law of cosines for angle at root
    cos_alpha = (l1 * l1 + clamped_dist * clamped_dist - l2 * l2) / (2.0 * l1 * clamped_dist)
    cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
    sin_alpha = float(np.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha)))

    # Orthonormal bend direction perpendicular to `forward`
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


class AnatomicalHandModel:
    """
    Articulated 3D human hand with 5 connected fingers (including an opposable thumb
    attached to the wrist and index knuckle via a fleshy thenar eminence web).
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
        palm_arch: float = 0.12,
        thumb_tip_target_world: Optional[np.ndarray] = None,
        index_tip_target_world: Optional[np.ndarray] = None,
        finger_spread: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[trimesh.Trimesh, str]]:
        """
        Builds a completely connected 3D human hand mesh in world coordinates.
        If `thumb_tip_target_world` or `index_tip_target_world` are provided, solves analytical IK
        in the hand's local frame so the thumb/index reach the target while remaining 100% attached
        to the wrist and palm knuckles.
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
            finger_curls = {
                "index": (0.65, 0.85, 0.50),
                "middle": (0.75, 0.95, 0.55),
                "ring": (0.85, 1.05, 0.60),
                "pinky": (0.95, 1.10, 0.65),
            }
        if finger_spread is None:
            finger_spread = {"index": 0.06, "middle": 0.0, "ring": -0.05, "pinky": -0.10}

        # 1. Wrist & Connected Deformable Metacarpal Palm
        wrist_pt = np.array([0.0, -1.5 * s, -28.0 * s])
        forearm_end = np.array([-3.0 * sign * s, -3.0 * s, -62.0 * s])
        parts.append((create_smooth_segment(wrist_pt, forearm_end, 17.5 * s, 20.5 * s, sections=16), "palm"))

        knuckle_x = {
            "index": 17.5 * sign * s,
            "middle": 6.0 * sign * s,
            "ring": -5.5 * sign * s,
            "pinky": -16.5 * sign * s,
        }
        knuckle_z = {
            "index": 25.0 * s,
            "middle": 27.5 * s,
            "ring": 25.5 * s,
            "pinky": 21.5 * s,
        }

        mcp_points: Dict[str, np.ndarray] = {}
        for fname in ["index", "middle", "ring", "pinky"]:
            kx = knuckle_x[fname]
            cup_y = abs(kx) * palm_arch
            p_mcp = np.array([kx, cup_y, knuckle_z[fname]])
            mcp_points[fname] = p_mcp
            # Metacarpal ray from wrist to knuckle
            parts.append((create_smooth_segment(wrist_pt, p_mcp, 12.5 * s, 8.2 * s, sections=14), "palm"))

        # Transverse metacarpal bridges & central palmar cushion (ensures a solid, continuous palm)
        for f_a, f_b in [("index", "middle"), ("middle", "ring"), ("ring", "pinky")]:
            parts.append((create_smooth_segment(mcp_points[f_a], mcp_points[f_b], 7.8 * s, 7.5 * s, sections=12), "palm"))

        p_palm_left = np.array([13.0 * sign * s, 1.5 * s, 0.0])
        p_palm_right = np.array([-13.0 * sign * s, 1.5 * s, -2.0 * s])
        parts.append((create_smooth_segment(p_palm_left, p_palm_right, 11.5 * s, 10.5 * s, sections=14), "palm"))

        # 2. Four Articulated Fingers (Index, Middle, Ring, Pinky)
        finger_dims = {
            "index": (25.0 * s, 16.5 * s, 12.5 * s, 6.4 * s),
            "middle": (27.5 * s, 18.0 * s, 13.0 * s, 6.7 * s),
            "ring": (25.5 * s, 17.0 * s, 12.0 * s, 6.2 * s),
            "pinky": (20.5 * s, 13.5 * s, 10.5 * s, 5.4 * s),
        }

        for fname, (l_prox, l_mid, l_dist, r_base) in finger_dims.items():
            p_mcp = mcp_points[fname]

            if fname == "index" and index_tip_target_world is not None:
                # Solve connected 3-phalanx IK from p_mcp to target_local
                target_local = world_to_local(index_tip_target_world)
                vec_tot = target_local - p_mcp
                dist_tot = float(np.linalg.norm(vec_tot))
                dir_tot = vec_tot / max(dist_tot, 1e-5)
                # Distal phalanx angles slightly inward toward palmar (+Y)
                dip_dir = dir_tot + np.array([0.0, 0.25, 0.0])
                dip_dir = dip_dir / max(float(np.linalg.norm(dip_dir)), 1e-5)
                p_dip_target = target_local - dip_dir * l_dist
                # Bend normal bows dorsally (-Y) and slightly outward (+Z)
                p_pip, p_dip = solve_two_bone_ik(
                    p_mcp, p_dip_target, l_prox, l_mid, bend_normal=np.array([0.0, -0.7, 0.7])
                )
                p_tip = target_local if dist_tot <= (l_prox + l_mid + l_dist) else (p_dip + dip_dir * l_dist)
            else:
                mcp_a, pip_a, dip_a = finger_curls.get(fname, (0.6, 0.8, 0.5))
                sp = finger_spread.get(fname, 0.0) * sign
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

            parts.append((create_smooth_segment(p_mcp, p_pip, r_base, r_base * 0.90, sections=14), "phalanx"))
            parts.append((create_smooth_segment(p_pip, p_dip, r_base * 0.90, r_base * 0.80, sections=14), "phalanx"))
            parts.append((create_smooth_segment(p_dip, p_tip, r_base * 0.80, r_base * 0.68, sections=14), "tip"))

        # 3. Opposable Thumb — 100% Connected to Wrist (CMC) and Index Knuckle (Thenar Web)
        if thumb_angles is None:
            thumb_angles = (0.65, 0.45, 0.35)
        th_abd, th_mcp, th_ip = thumb_angles

        # CMC joint anchored directly on radial side of wrist
        p_cmc = np.array([13.5 * sign * s, 2.5 * s, -18.0 * s])
        parts.append((create_smooth_segment(wrist_pt, p_cmc, 13.5 * s, 11.5 * s, sections=14), "palm"))

        l_thenar = 25.0 * s
        l_th_prox = 22.0 * s
        l_th_dist = 18.5 * s

        if thumb_tip_target_world is not None:
            th_target_local = world_to_local(thumb_tip_target_world)
            # Position p_thenar halfway toward target while anchored to p_cmc
            vec_c_t = th_target_local - p_cmc
            dist_c_t = float(np.linalg.norm(vec_c_t))
            dir_c_t = vec_c_t / max(dist_c_t, 1e-5)
            # Thenar biases slightly outward in radial X and palmar +Y
            thenar_bias = np.array([0.45 * sign, 0.35, 0.35])
            thenar_dir = 0.65 * dir_c_t + 0.35 * thenar_bias
            thenar_dir /= max(float(np.linalg.norm(thenar_dir)), 1e-5)
            p_thenar = p_cmc + thenar_dir * l_thenar

            # 2-bone IK for Thumb Proximal + Distal segments from p_thenar to th_target_local
            p_th_ip, p_th_tip = solve_two_bone_ik(
                p_thenar,
                th_target_local,
                l_th_prox,
                l_th_dist,
                bend_normal=np.array([0.6 * sign, 0.5, -0.4])
            )
        else:
            p_thenar = np.array([23.0 * sign * s, 9.5 * s, 4.0 * s])
            th_dir1 = np.array([
                sign * np.cos(th_abd) * 0.35,
                np.sin(th_abd),
                np.cos(th_mcp)
            ])
            th_dir1 /= np.linalg.norm(th_dir1)
            p_th_ip = p_thenar + th_dir1 * l_th_prox

            th_dir2 = np.array([
                -sign * np.sin(th_ip) * 0.45,
                np.sin(th_abd - th_ip * 0.4),
                np.cos(th_mcp + th_ip * 0.35)
            ])
            th_dir2 /= np.linalg.norm(th_dir2)
            p_th_tip = p_th_ip + th_dir2 * l_th_dist

        # Connect CMC -> Thenar (Thumb Metacarpal)
        parts.append((create_smooth_segment(p_cmc, p_thenar, 11.5 * s, 9.0 * s, sections=14), "thumb_base"))
        # Connect Thenar -> Index MCP Knuckle & Palm Center (Fleshy thumb-index web so thumb is NEVER detached!)
        parts.append((create_smooth_segment(p_thenar, mcp_points["index"], 9.2 * s, 8.0 * s, sections=12), "palm"))
        parts.append((create_smooth_segment(p_thenar, p_palm_left, 10.0 * s, 10.5 * s, sections=12), "palm"))

        # Connect Thenar -> Thumb IP -> Thumb Tip
        parts.append((create_smooth_segment(p_thenar, p_th_ip, 8.4 * s, 7.4 * s, sections=14), "thumb"))
        parts.append((create_smooth_segment(p_th_ip, p_th_tip, 7.4 * s, 6.4 * s, sections=14), "thumb_tip"))

        # Transform all parts into world coordinates
        T = np.eye(4)
        T[:3, :3] = palm_rot
        T[:3, 3] = palm_center

        transformed_parts: List[Tuple[trimesh.Trimesh, str]] = []
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
            "index": (0.5 * c, 0.9 * c, 0.6 * c),
            "middle": (0.6 * c, 1.0 * c, 0.6 * c),
            "ring": (0.7 * c, 1.1 * c, 0.7 * c),
            "pinky": (0.8 * c, 1.1 * c, 0.7 * c),
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


class DexterousRobotHandModel:
    """
    Anthropomorphic 5-Finger Dexterous Humanoid Robot Hand (modeled after Inspire / Unitree H1 / Optimus hands).
    Features a brushed titanium robotic palm chassis, 4 multi-phalanx mechanical fingers,
    a connected opposable robotic thumb with CMC rotary actuator, bronze joint pins, and cyan tactile pads.
    """

    def __init__(self, is_left_hand: bool = False, scale: float = 1.0):
        self.is_left = is_left_hand
        self.scale = scale
        self._human_kinematics = AnatomicalHandModel(is_left_hand=is_left_hand, scale=scale)

    def build_robotic_parts(
        self,
        palm_center: np.ndarray,
        palm_rot: np.ndarray,
        finger_curls: Optional[Dict[str, Tuple[float, float, float]]] = None,
        thumb_angles: Optional[Tuple[float, float, float]] = None,
        thumb_tip_target_world: Optional[np.ndarray] = None,
        index_tip_target_world: Optional[np.ndarray] = None,
        finger_spread: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[trimesh.Trimesh, str, float]]:
        """
        Returns a list of `(mesh, hex_color, alpha)` representing a 5-fingered robotic hand
        executing the exact same bimanual dexterity as the human hand twin.
        """
        base_parts = self._human_kinematics.build_articulated_parts(
            palm_center=palm_center,
            palm_rot=palm_rot,
            finger_curls=finger_curls,
            thumb_angles=thumb_angles,
            palm_arch=0.06,  # Rigid robotic chassis has lower arch
            thumb_tip_target_world=thumb_tip_target_world,
            index_tip_target_world=index_tip_target_world,
            finger_spread=finger_spread,
        )

        robot_meshes: List[Tuple[trimesh.Trimesh, str, float]] = []
        for mesh, ptype in base_parts:
            if ptype == "palm":
                # Dark anodized titanium robotic chassis
                robot_meshes.append((mesh, "#37474f", 0.98))
            elif ptype == "thumb_base":
                # Bronze/gold rotary CMC actuator housing
                robot_meshes.append((mesh, "#ffb300", 0.99))
            elif ptype in ("phalanx", "thumb"):
                # Brushed aluminum/silver mechanical phalanx links
                robot_meshes.append((mesh, "#90a4ae", 0.99))
            elif ptype in ("tip", "thumb_tip"):
                # High-friction cyan tactile sensor fingertip pads
                robot_meshes.append((mesh, "#00e5ff", 0.99))
            else:
                robot_meshes.append((mesh, "#607d8b", 0.98))

        return robot_meshes
