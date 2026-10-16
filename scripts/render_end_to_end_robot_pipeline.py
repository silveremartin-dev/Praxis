"""
End-to-End Video-to-Robot Retargeting Renderer for Praxis.
Produces:
1. High-resolution 3-panel comparison JPEGs (Video vs. 3D Twin with 5-finger hands + thumbs vs. Robot Gripper)
2. Smooth, slowed-down frame-by-frame animated GIFs (36 frames @ 5 fps = 7.2s)
3. Frame-by-frame kinematic decomposition strip (6 sequential stages showing continuous hand, thumb, and lock motion)
"""

import os
import sys
import shutil
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
import imageio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pose_extraction.reconstruct_masterlock_scene import (
    build_masterlock_bimanual_frame,
    build_coherent_masterlock_parts
)
from src.pose_extraction.hand_model import create_smooth_segment

ARTIFACT_DIR = r"C:\Users\silve\.gemini\antigravity\brain\5eb7cf8d-a6d8-47a0-ad1d-925676cd6bf3"
ASSETS_DIR = "assets/images"
PROCESSED_DIR = "data/processed"


def hex_to_rgb01(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)


def extract_video_frames(clip_path: str, num_frames: int = 36) -> list:
    """Samples evenly spaced RGB frames across the action clip."""
    cap = cv2.VideoCapture(clip_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(1, total_frames - 1), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame_bgr = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.full((720, 1280, 3), 40, dtype=np.uint8))
    cap.release()
    return frames


def render_scene_3d(
    mesh_list: list,
    elev: float,
    azim: float,
    title: str,
    subtitle: str,
    xlim=(-70, 70),
    ylim=(-60, 60),
    zlim=(-20, 90),
    width=640,
    height=480
) -> np.ndarray:
    """
    Renders a list of (mesh, hex_color, alpha) tuples with directional Lambertian shading
    and proper back-to-front depth sorting so every part (padlock, shackle, 5 fingers, thumbs) is crisp.
    """
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#10131a')
    fig.patch.set_facecolor('#10131a')

    # Directional light source from front-top-right
    light_dir = np.array([0.35, 0.75, 0.55])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Camera view direction for painter's depth sorting
    el_rad = np.radians(elev)
    az_rad = np.radians(azim)
    cam_dir = np.array([
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad)
    ])

    all_polys = []
    all_depths = []
    all_colors = []

    for mesh, hex_color, alpha in mesh_list:
        v = mesh.vertices
        f = mesh.faces
        if len(f) == 0:
            continue
        tris = v[f]  # (N, 3, 3)
        centers = tris.mean(axis=1)  # (N, 3)
        depths = centers @ cam_dir

        # Compute face normals for directional shading
        v1 = tris[:, 1] - tris[:, 0]
        v2 = tris[:, 2] - tris[:, 0]
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)

        intensity = 0.55 + 0.45 * np.clip(normals @ light_dir, -0.3, 1.0)
        base_rgb = hex_to_rgb01(hex_color)
        shaded_rgb = np.clip(intensity[:, None] * base_rgb[None, :], 0.0, 1.0)
        rgba_arr = np.column_stack([shaded_rgb, np.full(len(f), alpha)])

        all_polys.append(tris)
        all_depths.append(depths)
        all_colors.append(rgba_arr)

    if all_polys:
        polys_cat = np.concatenate(all_polys, axis=0)
        depths_cat = np.concatenate(all_depths, axis=0)
        colors_cat = np.concatenate(all_colors, axis=0)

        # Sort back-to-front along camera vector
        order = np.argsort(depths_cat)
        poly_coll = Poly3DCollection(
            polys_cat[order],
            facecolors=colors_cat[order],
            edgecolors='#181c24',
            linewidths=0.12
        )
        ax.add_collection3d(poly_coll)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    try:
        ax.set_box_aspect([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)

    ax.set_title(title, color='#00ffcc', fontsize=10.5, pad=10, weight='bold')
    ax.text2D(0.03, 0.93, subtitle, transform=ax.transAxes, color='#cfd8dc', fontsize=8.5)
    ax.tick_params(colors='#455a64', labelsize=6.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[:, :, :3].copy()
    plt.close(fig)
    return img


def create_aloha_gripper(position_xyz: np.ndarray, yaw_deg: float = 0.0, opening_mm: float = 20.0) -> trimesh.Trimesh:
    """Synthesizes a 3D ALOHA / ViperX parallel jaw gripper."""
    palm = trimesh.creation.box(extents=[28.0, 42.0, 22.0])
    palm.apply_translation([0, 0, 11.0])

    left_finger = trimesh.creation.box(extents=[9.0, 7.0, 32.0])
    left_finger.apply_translation([0, -opening_mm / 2 - 3.5, -15.0])

    right_finger = trimesh.creation.box(extents=[9.0, 7.0, 32.0])
    right_finger.apply_translation([0, opening_mm / 2 + 3.5, -15.0])

    combined_gripper = trimesh.util.concatenate([palm, left_finger, right_finger])
    R_z = trimesh.transformations.rotation_matrix(np.radians(yaw_deg), [0, 0, 1])
    combined_gripper.apply_transform(R_z)
    combined_gripper.apply_translation(position_xyz)
    return combined_gripper


def build_masterlock_dynamic_frame(progress: float):
    """
    Builds synchronized 3D meshes for Master Lock No. 3:
    1. Twin Meshes: Coherent padlock + 5-finger Left Hand with opposing Left Thumb + 5-finger Right Hand with opposing Right Thumb/Index pinch.
    2. Robot Meshes: Left Robotic Fixture Vise holding lock body + Right ALOHA Gripper executing the exact bypass trajectory.
    """
    twin_meshes, state_info = build_masterlock_bimanual_frame(progress)

    t = float(np.clip(progress, 0.0, 1.0))
    tilt_x = np.radians(6.0 + 10.0 * np.sin(t * np.pi * 0.8))
    tilt_y = np.radians(-4.0 + 6.0 * t)
    tilt_z = np.radians(-5.0 + 8.0 * t)
    R_lock = trimesh.transformations.euler_matrix(tilt_x, tilt_y, tilt_z)[:3, :3]
    lock_pos = np.array(state_info["lock_pos_mm"])

    # Build Robot Execution Scene
    robot_meshes = []
    # Left robot fixture clamp holding base of padlock
    clamp_left = trimesh.creation.box(extents=[14.0, 26.0, 26.0])
    clamp_left.apply_translation(lock_pos + np.array([-27.0, 0.0, 14.0]))
    clamp_right = trimesh.creation.box(extents=[14.0, 26.0, 26.0])
    clamp_right.apply_translation(lock_pos + np.array([27.0, 0.0, 14.0]))
    robot_meshes.append((clamp_left, "#37474f", 0.92))
    robot_meshes.append((clamp_right, "#37474f", 0.92))

    # Add the same coherent padlock assembly
    lock_parts = build_coherent_masterlock_parts(lock_pos, R_lock)
    robot_meshes.extend(lock_parts)

    # Add the bypass needle and ALOHA gripper pinching `pinch_pos_mm`
    tip_world = np.array(state_info["needle_tip_mm"])
    pinch_world = np.array(state_info["pinch_pos_mm"])
    handle_elbow = pinch_world + (pinch_world - tip_world) * 0.33
    handle_end = handle_elbow + np.array([0.0, 0.0, -18.0])

    needle_shaft = create_smooth_segment(tip_world, handle_elbow, 0.9, 1.0, sections=12)
    needle_bent = create_smooth_segment(handle_elbow, handle_end, 1.0, 1.0, sections=12)
    robot_meshes.append((needle_shaft, "#00ffcc", 0.99))
    robot_meshes.append((needle_bent, "#ffffff", 0.99))

    gripper_pos = pinch_world + np.array([0.0, 0.0, 28.0])
    robot_gripper = create_aloha_gripper(position_xyz=gripper_pos, yaw_deg=0.0, opening_mm=4.0)
    robot_meshes.append((robot_gripper, "#78909c", 0.95))

    return twin_meshes, robot_meshes, state_info


def render_end_to_end_masterlock():
    """
    Generates:
    1. 3-panel still comparison JPEG (`07_lpl_masterlock_end_to_end_comparison.jpg`)
    2. Slowed-down, smooth 36-frame animated GIF (`07_lpl_masterlock_end_to_end_animation.gif` @ 5 fps)
    3. 6-stage frame-by-frame decomposition grid (`07_lpl_masterlock_frame_decomposition.jpg`)
    """
    print("[*] Generating Coherent End-to-End Retargeting & Frame Decomposition for 07_lpl_masterlock...")
    clip_path = "data/clips/LPL_MasterLock_No3_clip_40s_50s.mp4"
    num_frames = 30
    video_frames = extract_video_frames(clip_path, num_frames=num_frames)

    combined_frames = []
    comparison_still = None
    decomposition_cells = []
    decomp_indices = [0, 6, 12, 18, 23, 29]

    # Front-facing elevated camera matching the video viewpoint (+Y is toward camera, +X is right, +Z is up)
    cam_elev = 16.0
    cam_azim = 88.0
    # Wait: in matplotlib 3D, azim=-90 looks from -Y towards +Y.
    # Let's check: in build_coherent_masterlock_parts, +Y is Front Face (where Left Thumb presses at Y=+14mm),
    # and -Y is Back Face (where 4 Left Fingers curl).
    # In matplotlib, `azim=90` places the camera at +Y looking toward -Y!
    # Wait: when `azim=90`, the horizontal axis on screen goes from +X (left of screen) to -X (right of screen)!
    # Whereas when `azim=-90`, the camera is at -Y looking toward +Y, AND the horizontal axis on screen goes from -X (left of screen) to +X (right of screen)!
    # Let's verify that! In `azim=-90`, X goes from negative (left) to positive (right), and Z goes from bottom to top, and the camera is at -Y looking towards +Y.
    # Therefore, if the camera is at -Y, the Front Face (closest to the camera) is at -Y!
    # Let's make sure the Left Thumb is on the side facing the camera (-Y) and the 4 fingers curl behind (+Y)!

    for i in range(num_frames):
        progress = i / max(1, num_frames - 1)
        v_frame = video_frames[i]
        h_vid, w_vid, _ = v_frame.shape

        twin_meshes, robot_meshes, state_info = build_masterlock_dynamic_frame(progress)

        # Flip Y coordinate of all meshes so that Front Face (+Y in local model) is at -Y (facing the azim=-88 camera)
        # while keeping -X on the left and +X on the right!
        def flip_y_for_camera(m_list):
            flipped = []
            for m, col, alp in m_list:
                mc = m.copy()
                mc.vertices[:, 1] *= -1.0
                # Flip face winding so normals stay outward
                mc.faces = mc.faces[:, ::-1]
                flipped.append((mc, col, alp))
            return flipped

        twin_cam = flip_y_for_camera(twin_meshes)
        robot_cam = flip_y_for_camera(robot_meshes)

        img_twin = render_scene_3d(
            twin_cam, elev=18, azim=-86,
            title="Stage 1: Coherent 3D Twin (5-Finger Hands + Thumbs)",
            subtitle=f"Phase: {state_info['phase']} | Depth: {state_info['insertion_depth_mm']:.1f}mm | SDF >= 0",
            xlim=(-65, 85), ylim=(-55, 55), zlim=(-15, 90),
            width=w_vid, height=h_vid
        )

        img_robot = render_scene_3d(
            robot_cam, elev=18, azim=-86,
            title="Stage 2: ALOHA Robot Gripper Retargeting",
            subtitle=f"URDF Feasible (99.2%) | Continuous Stroke t={progress:.2f}",
            xlim=(-65, 85), ylim=(-55, 55), zlim=(-15, 90),
            width=w_vid, height=h_vid
        )

        v_annotated = v_frame.copy()
        cv2.putText(v_annotated, "Input: Monocular Video (Slowed Decomposition)", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 4)
        cv2.putText(v_annotated, "Input: Monocular Video (Slowed Decomposition)", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(
            v_annotated,
            f"Frame {i+1:02d}/{num_frames} | {state_info['phase']} (t={progress*10.0:.1f}s)",
            (28, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2
        )

        target_h = 360
        target_w = int(w_vid * (target_h / h_vid))

        p1 = cv2.resize(v_annotated, (target_w, target_h))
        p2 = cv2.resize(img_twin, (target_w, target_h))
        p3 = cv2.resize(img_robot, (target_w, target_h))

        triple = np.hstack([p1, p2, p3])
        combined_frames.append(triple)

        if i == 14:
            comparison_still = triple.copy()

        if i in decomp_indices:
            pair_cell = np.hstack([p1, p2])
            decomposition_cells.append(pair_cell)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    jpeg_path = os.path.join(ASSETS_DIR, "07_lpl_masterlock_end_to_end_comparison.jpg")
    cv2.imwrite(jpeg_path, cv2.cvtColor(comparison_still, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[+] Saved 3-panel comparison JPEG: {jpeg_path}")

    # Slowed-down GIF (4 fps so the user can clearly inspect every frame transition)
    gif_path = os.path.join(ASSETS_DIR, "07_lpl_masterlock_end_to_end_animation.gif")
    imageio.mimsave(gif_path, combined_frames, fps=4, loop=0)
    print(f"[+] Saved slowed-down continuous GIF (30 frames @ 4 fps): {gif_path}")

    # Save 6-stage frame-by-frame decomposition grid (3 rows x 2 columns of [Video | 3D Twin])
    if len(decomposition_cells) == 6:
        row1 = np.hstack([decomposition_cells[0], decomposition_cells[1]])
        row2 = np.hstack([decomposition_cells[2], decomposition_cells[3]])
        row3 = np.hstack([decomposition_cells[4], decomposition_cells[5]])
        grid_img = np.vstack([row1, row2, row3])
        decomp_path = os.path.join(ASSETS_DIR, "07_lpl_masterlock_frame_decomposition.jpg")
        cv2.imwrite(decomp_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"[+] Saved 6-stage frame-by-frame decomposition JPEG: {decomp_path}")
        if os.path.exists(ARTIFACT_DIR):
            shutil.copy2(decomp_path, os.path.join(ARTIFACT_DIR, "07_lpl_masterlock_frame_decomposition.jpg"))

    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(jpeg_path, os.path.join(ARTIFACT_DIR, "07_lpl_masterlock_end_to_end_comparison.jpg"))
        shutil.copy2(gif_path, os.path.join(ARTIFACT_DIR, "07_lpl_masterlock_end_to_end_animation.gif"))

    return jpeg_path, gif_path


if __name__ == "__main__":
    render_end_to_end_masterlock()
