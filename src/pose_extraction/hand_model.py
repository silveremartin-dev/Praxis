"""
Anatomical 3D Hand Mesh Generator for Praxis.
Constructs realistic multi-segment articulated human hand meshes (wrist, palm, thumb, and 4 fingers)
complying with anthropometric dimensions for bimanual manipulation grounding.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh

def create_capsule(radius: float, height: float, sections: int = 12) -> trimesh.Trimesh:
    """Creates a smooth capsule primitive along the Z axis with moderate face count."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    s_top = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    s_top.apply_translation([0, 0, height / 2])
    s_bot = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    s_bot.apply_translation([0, 0, -height / 2])
    return trimesh.util.concatenate([cyl, s_top, s_bot])

class AnatomicalHandModel:
    """
    Constructs an articulated 3D hand mesh with realistic palm, thumb, and fingers.
    """
    def __init__(self, is_left_hand: bool = False, scale: float = 1.0):
        self.is_left = is_left_hand
        self.scale = scale

    def build_hand_mesh(
        self,
        wrist_pos: np.ndarray,
        wrist_rot_matrix: np.ndarray,
        finger_curl: float = 0.5, # 0.0 = flat open, 1.0 = fully curled fist
        thumb_oppose: float = 0.6,
        skin_color: Tuple[int, int, int, int] = (235, 175, 155, 255)
    ) -> trimesh.Trimesh:
        """
        Builds a complete articulated hand mesh oriented in space.
        """
        meshes = []
        sign = -1.0 if self.is_left else 1.0

        # 1. Palm / Metacarpal body (width 60mm, thickness 18mm, length 55mm)
        palm = trimesh.creation.box(extents=[58.0 * self.scale, 18.0 * self.scale, 52.0 * self.scale])
        palm.apply_translation([0, 0, 26.0 * self.scale])
        meshes.append(palm)

        # 2. Wrist / Forearm base
        wrist = trimesh.creation.cylinder(radius=20.0 * self.scale, height=30.0 * self.scale, sections=16)
        wrist.apply_translation([0, 0, -15.0 * self.scale])
        meshes.append(wrist)

        # 3. Four Fingers (Index, Middle, Ring, Pinky)
        finger_specs = [
            ("index", 17.0 * sign, 38.0, 6.5),
            ("middle", 6.0 * sign, 42.0, 7.0),
            ("ring", -5.0 * sign, 39.0, 6.5),
            ("pinky", -16.0 * sign, 32.0, 6.0),
        ]

        curl_angle = finger_curl * (np.pi * 0.55) # 0 to ~100 degrees curl

        for name, x_offset, length, radius in finger_specs:
            r = radius * self.scale
            l_prox = length * 0.50 * self.scale
            l_dist = length * 0.50 * self.scale

            # Proximal phalanx (angled forward/down)
            prox = create_capsule(radius=r, height=l_prox, sections=12)
            R_curl1 = trimesh.transformations.rotation_matrix(curl_angle * 0.6, [1, 0, 0])
            prox.apply_transform(R_curl1)
            prox.apply_translation([x_offset * self.scale, 0, 52.0 * self.scale + l_prox / 2])
            
            # Intermediate + Distal phalanx (curled inwards)
            dist = create_capsule(radius=r * 0.85, height=l_dist, sections=12)
            R_curl2 = trimesh.transformations.rotation_matrix(curl_angle * 1.1, [1, 0, 0])
            dist.apply_transform(R_curl2)
            tip_offset_y = np.sin(curl_angle * 0.6) * l_prox
            tip_offset_z = np.cos(curl_angle * 0.6) * l_prox
            dist.apply_translation([x_offset * self.scale, tip_offset_y, 52.0 * self.scale + tip_offset_z + l_dist / 2])

            meshes.extend([prox, dist])

        # 4. Thumb (Opposable kinematic chain)
        thumb_len = 34.0 * self.scale
        thumb_r = 7.5 * self.scale
        
        # Proximal metacarpal
        thumb_prox = create_capsule(radius=thumb_r, height=thumb_len * 0.5, sections=12)
        thumb_angle_y = sign * np.radians(40 + thumb_oppose * 30)
        thumb_angle_x = np.radians(35 + thumb_oppose * 35)
        R_th1 = trimesh.transformations.euler_matrix(thumb_angle_x, thumb_angle_y, 0)
        thumb_prox.apply_transform(R_th1)
        thumb_prox.apply_translation([sign * 26.0 * self.scale, 8.0 * self.scale, 22.0 * self.scale])

        # Distal phalanx
        thumb_dist = create_capsule(radius=thumb_r * 0.85, height=thumb_len * 0.5, sections=12)
        R_th2 = trimesh.transformations.euler_matrix(thumb_angle_x + np.radians(30), thumb_angle_y, 0)
        thumb_dist.apply_transform(R_th2)
        thumb_dist.apply_translation([sign * 32.0 * self.scale, 15.0 * self.scale, 35.0 * self.scale])

        meshes.extend([thumb_prox, thumb_dist])

        # Combine all hand parts into a single mesh
        combined_hand = trimesh.util.concatenate(meshes)
        
        # Apply global wrist transformation
        T = np.eye(4)
        T[:3, :3] = wrist_rot_matrix
        T[:3, 3] = wrist_pos
        combined_hand.apply_transform(T)
        combined_hand.visual.face_colors = list(skin_color)
        return combined_hand
