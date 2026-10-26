"""
Generic Parametric 3D CAD Primitive & Compound Assembly Builder for Praxis.
Constructs rigid and articulated 3D workpieces and tools from declarative specifications
without requiring video-specific source code.
"""

from typing import Dict, List, Tuple
import numpy as np
import trimesh
from src.pose_extraction.hand_model import create_smooth_capsule


def create_cylinder(
    radius: float,
    height: float,
    sections: int = 24,
    color: Tuple[int, int, int, int] = (180, 190, 200, 255)
) -> trimesh.Trimesh:
    """Creates a metric cylinder primitive."""
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    m.visual.face_colors = list(color)
    return m


def create_box(
    extents: List[float],
    color: Tuple[int, int, int, int] = (160, 170, 180, 255)
) -> trimesh.Trimesh:
    """Creates a metric box primitive."""
    m = trimesh.creation.box(extents=extents)
    m.visual.face_colors = list(color)
    return m


class ParametricCADBuilder:
    """
    Builds coherent 3D multi-part assemblies (padlocks, articulated keys, watch cases,
    cast-iron plates, precision probes) from declarative parameters.
    """

    @staticmethod
    def build_laminated_padlock(
        pos: np.ndarray,
        rot: np.ndarray,
        width: float = 38.0,
        thickness: float = 20.0,
        height: float = 32.0,
        shackle_open_left: bool = True
    ) -> List[Tuple[trimesh.Trimesh, str, float]]:
        """
        Builds a realistic contoured Master Lock No. 3 assembly transformed rigidly by (pos, rot).
        Matches the real physical geometry visible in the video:
        - Not a flat parallelepiped: features a proud/bulged central face (20.0mm thick across X in [-10.5, +10.5]mm),
          stepped intermediate shoulders (16.8mm thick), and narrower outer rivet wings (13.6mm thick across X in [-19, +19]mm).
        - 7.5mm lower red vinyl 'Master' bumper following the contoured profile.
        - 7mm diameter hardened steel shackle swung 180° open to the left.
        - Single brass key inserted vertically upward into the bottom keyway (no spurious horizontal key).
        """
        raw_parts: List[Tuple[trimesh.Trimesh, str, float]] = []

        wing_thickness = thickness * 0.68      # 13.6mm at left/right rivet wings
        shoulder_thickness = thickness * 0.84  # 16.8mm at beveled transition shoulders
        center_width = width * 0.55            # 20.9mm proud central face
        shoulder_width = width * 0.76          # 28.9mm intermediate shoulder span

        # 1. Lower Contoured Red Vinyl Bumper Band (Z = 0 to 7.5mm)
        bumper_wings = trimesh.creation.box(extents=[width + 0.8, wing_thickness + 0.8, 7.5])
        bumper_wings.apply_translation([0.0, 0.0, 3.75])
        raw_parts.append((bumper_wings, "#c62828", 0.99))

        bumper_shoulder = trimesh.creation.box(extents=[shoulder_width + 0.8, shoulder_thickness + 0.8, 7.5])
        bumper_shoulder.apply_translation([0.0, 0.0, 3.75])
        raw_parts.append((bumper_shoulder, "#d32f2f", 0.99))

        bumper_center = trimesh.creation.box(extents=[center_width + 0.8, thickness + 0.8, 7.5])
        bumper_center.apply_translation([0.0, 0.0, 3.75])
        raw_parts.append((bumper_center, "#e53935", 0.99))

        # 2. Contoured Laminated Steel Core (Z = 7.5 to height=32mm)
        core_h = height - 7.5
        core_z = 7.5 + core_h / 2.0

        # Outer rivet wings (narrower in Y)
        core_wings = trimesh.creation.box(extents=[width, wing_thickness, core_h])
        core_wings.apply_translation([0.0, 0.0, core_z])
        raw_parts.append((core_wings, "#90a4ae", 0.99))

        # Intermediate beveled shoulder
        core_shoulder = trimesh.creation.box(extents=[shoulder_width, shoulder_thickness, core_h])
        core_shoulder.apply_translation([0.0, 0.0, core_z])
        raw_parts.append((core_shoulder, "#b0bec5", 0.99))

        # Proud central face (bulges out in the middle of the front & back faces)
        core_center = trimesh.creation.box(extents=[center_width, thickness, core_h])
        core_center.apply_translation([0.0, 0.0, core_z])
        raw_parts.append((core_center, "#cfd8dc", 0.99))

        # Horizontal lamination plate ribs across the contoured body
        for z_rib in np.linspace(9.5, height - 2.0, 6):
            rib_c = trimesh.creation.box(extents=[center_width + 0.3, thickness + 0.3, 1.4])
            rib_c.apply_translation([0.0, 0.0, float(z_rib)])
            raw_parts.append((rib_c, "#eceff1", 0.99))
            rib_w = trimesh.creation.box(extents=[width + 0.3, wing_thickness + 0.3, 1.4])
            rib_w.apply_translation([0.0, 0.0, float(z_rib)])
            raw_parts.append((rib_w, "#b0bec5", 0.99))

        # 3. Four Corner Rivet Heads on Top Plate (seated on the outer rivet wings)
        for rx in [-width * 0.40, width * 0.40]:
            for ry in [-wing_thickness * 0.32, wing_thickness * 0.32]:
                rivet = trimesh.creation.cylinder(radius=1.6, height=1.2, sections=12)
                rivet.apply_translation([rx, ry, height + 0.4])
                raw_parts.append((rivet, "#78909c", 0.99))

        # 4. Open Shackle Cavity on Top Right (circular depression at X = +10.5mm)
        hole_x = width * 0.275  # 10.45mm
        hole_cavity = trimesh.creation.cylinder(radius=3.8, height=1.5, sections=16)
        hole_cavity.apply_translation([hole_x, 0.0, height + 0.2])
        raw_parts.append((hole_cavity, "#1a1d24", 0.99))

        # 5. Hardened Steel Shackle (7mm diameter, swung 180° to the left)
        shackle_r = 3.4
        pivot_x = -hole_x  # -10.45mm
        p_pivot_base = np.array([pivot_x, 0.0, height - 1.0])
        p_pivot_top = np.array([pivot_x, 0.0, height + 21.0])
        raw_parts.append((create_smooth_capsule(p_pivot_base, p_pivot_top, shackle_r, shackle_r, 16), "#eceff1", 0.99))

        # Semicircular arch connecting X = -10.45mm to X = -31.35mm
        arch_span = -2.0 * hole_x if shackle_open_left else 2.0 * hole_x
        arch_center_x = pivot_x + 0.5 * arch_span
        arch_radius = abs(0.5 * arch_span)
        angles = np.linspace(0.0, np.pi, 13)
        for k in range(12):
            a0, a1 = angles[k], angles[k + 1]
            p0 = np.array([arch_center_x + arch_radius * np.cos(a0), 0.0, height + 21.0 + arch_radius * np.sin(a0)])
            p1 = np.array([arch_center_x + arch_radius * np.cos(a1), 0.0, height + 21.0 + arch_radius * np.sin(a1)])
            raw_parts.append((create_smooth_capsule(p0, p1, shackle_r, shackle_r, 14), "#f5f7fa", 0.99))

        # Open toe leg with locking notch hanging free on the left
        toe_x = pivot_x + arch_span
        p_toe_top = np.array([toe_x, 0.0, height + 21.0])
        p_toe_bot = np.array([toe_x, 0.0, height + 8.0])
        raw_parts.append((create_smooth_capsule(p_toe_top, p_toe_bot, shackle_r, shackle_r * 0.92, 16), "#eceff1", 0.99))

        # Locking notch cut into inner side of toe
        notch = trimesh.creation.cylinder(radius=1.9, height=3.2, sections=12)
        notch.apply_translation([toe_x + 2.4, 0.0, height + 14.0])
        raw_parts.append((notch, "#78909c", 0.99))

        # 6. Single Legitimate Brass Key Inserted Vertically into Bottom Keyway
        # (Blade is inside the vertical cylinder; only the brass bow and collar hang below Z=0)
        key_bow = trimesh.creation.cylinder(radius=7.5, height=2.0, sections=16)
        R_k = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        key_bow.apply_transform(R_k)
        key_bow.apply_translation([3.0, 0.0, -7.0])
        raw_parts.append((key_bow, "#d4af37", 0.98))

        key_neck = trimesh.creation.box(extents=[5.0, 2.0, 5.0])
        key_neck.apply_translation([3.0, 0.0, -2.0])
        raw_parts.append((key_neck, "#e6c229", 0.98))

        # Apply global 6-DoF transformation
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos

        transformed: List[Tuple[trimesh.Trimesh, str, float]] = []
        for mesh, color, alpha in raw_parts:
            m = mesh.copy()
            m.apply_transform(T)
            transformed.append((m, color, alpha))
        return transformed

    @staticmethod
    def build_l_shaped_probe(
        tip_world: np.ndarray,
        pinch_world: np.ndarray,
        handle_drop_mm: float = 16.0
    ) -> List[Tuple[trimesh.Trimesh, str, float]]:
        """
        Builds a flat spring steel bypass decoder needle (0.5mm x 1.5mm) from tip through pinch.
        """
        tip_world = np.asarray(tip_world, dtype=float)
        pinch_world = np.asarray(pinch_world, dtype=float)
        shaft_vec = pinch_world - tip_world
        shaft_len = float(np.linalg.norm(shaft_vec))
        shaft_dir = shaft_vec / max(shaft_len, 1e-5)

        handle_elbow = pinch_world + shaft_dir * 14.0
        handle_end = handle_elbow + np.array([2.5, 0.0, -handle_drop_mm])

        shaft = create_smooth_capsule(tip_world, handle_elbow, 0.95, 1.10, sections=12)
        bent = create_smooth_capsule(handle_elbow, handle_end, 1.10, 1.10, sections=12)
        return [
            (shaft, "#00e5ff", 1.0),
            (bent, "#ffffff", 1.0),
        ]
