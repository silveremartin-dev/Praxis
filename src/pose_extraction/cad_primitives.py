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
        width: float = 44.0,
        thickness: float = 22.0,
        height: float = 38.0,
        shackle_open_left: bool = True
    ) -> List[Tuple[trimesh.Trimesh, str, float]]:
        """
        Builds a coherent laminated steel padlock assembly transformed rigidly by (pos, rot).
        Dimensions match Master Lock Commercial No. 3:
        - 44mm wide x 22mm thick x 38mm high body
        - 7.5mm diameter hardened steel shackle with 180° swing
        - 9mm lower red vinyl bumper
        """
        raw_parts: List[Tuple[trimesh.Trimesh, str, float]] = []

        # 1. Lower Red Vinyl Bumper Band (Z = 0 to 9mm)
        bumper = trimesh.creation.box(extents=[width + 1.4, thickness + 1.2, 9.0])
        bumper.apply_translation([0.0, 0.0, 4.5])
        raw_parts.append((bumper, "#d32f2f", 0.99))

        # 2. Laminated Steel Core (Z = 9 to height=38mm)
        core_h = height - 9.0
        main_core = trimesh.creation.box(extents=[width - 0.4, thickness - 0.2, core_h])
        main_core.apply_translation([0.0, 0.0, 9.0 + core_h / 2.0])
        raw_parts.append((main_core, "#b0bec5", 0.99))

        # Horizontal lamination ribs for realistic laminated plate texture
        for z_rib in np.linspace(11.5, height - 2.5, 7):
            rib = trimesh.creation.box(extents=[width + 0.3, thickness + 0.3, 2.0])
            rib.apply_translation([0.0, 0.0, float(z_rib)])
            raw_parts.append((rib, "#cfd8dc", 0.99))

        # 3. Four Corner Rivet Heads on Top Plate
        for rx in [-width * 0.38, width * 0.38]:
            for ry in [-thickness * 0.33, thickness * 0.33]:
                rivet = trimesh.creation.cylinder(radius=1.8, height=1.4, sections=12)
                rivet.apply_translation([rx, ry, height + 0.5])
                raw_parts.append((rivet, "#90a4ae", 0.99))

        # 4. Open Shackle Cavity on Top Right (circular depression at X = +11mm)
        hole_x = width * 0.275  # 11.0mm
        hole_cavity = trimesh.creation.cylinder(radius=4.0, height=1.5, sections=16)
        hole_cavity.apply_translation([hole_x, 0.0, height + 0.2])
        raw_parts.append((hole_cavity, "#1a1d24", 0.99))

        # 5. Hardened Steel Shackle (7mm diameter, swung 180° to the left)
        shackle_r = 3.5
        pivot_x = -hole_x  # -11.0mm
        p_pivot_base = np.array([pivot_x, 0.0, height - 1.0])
        p_pivot_top = np.array([pivot_x, 0.0, height + 23.0])
        raw_parts.append((create_smooth_capsule(p_pivot_base, p_pivot_top, shackle_r, shackle_r, 16), "#eceff1", 0.99))

        # Semicircular arch connecting X = -11.0mm to X = -33.0mm
        arch_span = -2.0 * hole_x if shackle_open_left else 2.0 * hole_x
        arch_center_x = pivot_x + 0.5 * arch_span
        arch_radius = abs(0.5 * arch_span)
        angles = np.linspace(0.0, np.pi, 13)
        for k in range(12):
            a0, a1 = angles[k], angles[k + 1]
            p0 = np.array([arch_center_x + arch_radius * np.cos(a0), 0.0, height + 23.0 + arch_radius * np.sin(a0)])
            p1 = np.array([arch_center_x + arch_radius * np.cos(a1), 0.0, height + 23.0 + arch_radius * np.sin(a1)])
            raw_parts.append((create_smooth_capsule(p0, p1, shackle_r, shackle_r, 14), "#f5f7fa", 0.99))

        # Open toe leg with locking notch hanging free on the left
        toe_x = pivot_x + arch_span  # -33.0mm
        p_toe_top = np.array([toe_x, 0.0, height + 23.0])
        p_toe_bot = np.array([toe_x, 0.0, height + 9.0])
        raw_parts.append((create_smooth_capsule(p_toe_top, p_toe_bot, shackle_r, shackle_r * 0.92, 16), "#eceff1", 0.99))

        # Locking notch cut into inner side of toe
        notch = trimesh.creation.cylinder(radius=2.0, height=3.5, sections=12)
        notch.apply_translation([toe_x + 2.5, 0.0, height + 15.0])
        raw_parts.append((notch, "#78909c", 0.99))

        # 6. Bottom Brass Key in Cylinder Keyway
        key_bow = trimesh.creation.cylinder(radius=8.5, height=2.0, sections=16)
        R_k = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        key_bow.apply_transform(R_k)
        key_bow.apply_translation([8.0, 2.0, -5.5])
        raw_parts.append((key_bow, "#d4af37", 0.96))

        key_blade = trimesh.creation.box(extents=[22.0, 1.8, 5.0])
        key_blade.apply_translation([20.0, 2.0, 1.0])
        raw_parts.append((key_blade, "#e6c229", 0.96))

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
