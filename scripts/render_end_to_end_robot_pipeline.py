"""
End-to-End Video-to-Dexterous-Robot Retargeting Renderer for Praxis.
Uses the generic `VideoPoseExtractor` (`src/pose_extraction/extractor.py`),
`AnatomicalHandModel`, and `DexterousRobotHandModel` (`src/pose_extraction/hand_model.py`).

Produces:
1. High-resolution 3-panel comparison JPEG:
   [Panel 1: Monocular Video | Panel 2: Connected 5-Finger Human Twin | Panel 3: Bimanual 5-Finger Dexterous Robot Hands]
2. Slowed-down continuous 30-frame animated GIF (4 fps = 7.5s) covering all 5 gesture phases
   (approach, horizontal collar slide, withdrawal, needle drop & left thumb slide, and right index pointing).
3. 6-stage frame-by-frame decomposition grid (`07_lpl_masterlock_frame_decomposition.jpg`).
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
import imageio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pose_extraction.extractor import VideoPoseExtractor
from tests.pose_extraction.reconstruct_masterlock_scene import build_masterlock_scene

ARTIFACT_DIR = r"C:\Users\silve\.gemini\antigravity\brain\5eb7cf8d-a6d8-47a0-ad1d-925676cd6bf3"
ASSETS_DIR = "assets/images"
PROCESSED_DIR = "data/processed"


def hex_to_rgb01(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)


def extract_video_frames(clip_path: str, num_frames: int = 30) -> list:
    """Samples evenly spaced RGB frames across the action clip."""
    cap = cv2.VideoCapture(clip_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(1, total_frames - 1), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
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
    xlim=(-70, 75),
    ylim=(-55, 55),
    zlim=(-22, 85),
    width=640,
    height=480
) -> np.ndarray:
    """
    Renders a list of (mesh, hex_color, alpha) tuples with directional Lambertian shading
    and painter's algorithm back-to-front depth sorting.
    """
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#10131a')
    fig.patch.set_facecolor('#10131a')

    light_dir = np.array([0.30, -0.75, 0.58])
    light_dir = light_dir / np.linalg.norm(light_dir)

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
        tris = v[f]
        centers = tris.mean(axis=1)
        depths = centers @ cam_dir

        v1 = tris[:, 1] - tris[:, 0]
        v2 = tris[:, 2] - tris[:, 0]
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-8)

        intensity = 0.56 + 0.44 * np.clip(normals @ light_dir, -0.25, 1.0)
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

        order = np.argsort(depths_cat)
        poly_coll = Poly3DCollection(
            polys_cat[order],
            facecolors=colors_cat[order],
            edgecolors='#181c24',
            linewidths=0.10
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

    ax.set_title(title, color='#00ffcc', fontsize=10.0, pad=8, weight='bold')
    ax.text2D(0.03, 0.93, subtitle, transform=ax.transAxes, color='#cfd8dc', fontsize=8.2)
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


def flip_y_for_camera(m_list: list) -> list:
    """
    Maps local model +Y (front face toward camera) to -Y in matplotlib 3D coordinates
    so that `azim=-88` looks directly at the front face with -X on the left and +X on the right.
    """
    flipped = []
    for m, col, alp in m_list:
        mc = m.copy()
        mc.vertices[:, 1] *= -1.0
        mc.faces = mc.faces[:, ::-1]
        flipped.append((mc, col, alp))
    return flipped


def render_end_to_end_masterlock():
    """
    Generates:
    1. 3-panel still comparison JPEG (`07_lpl_masterlock_end_to_end_comparison.jpg`)
    2. Slowed-down, smooth 30-frame animated GIF (`07_lpl_masterlock_end_to_end_animation.gif` @ 4 fps)
    3. 6-stage frame-by-frame decomposition grid (`07_lpl_masterlock_frame_decomposition.jpg`)
    """
    print("[*] Updating 3D OBJ & Metadata via generic VideoPoseExtractor...")
    build_masterlock_scene(output_dir=PROCESSED_DIR, progress=0.30)

    print("[*] Generating End-to-End Bimanual Human & 5-Finger Robot Hand Renders...")
    clip_path = "data/clips/LPL_MasterLock_No3_clip_40s_50s.mp4"
    num_frames = 30
    video_frames = extract_video_frames(clip_path, num_frames=num_frames)
    extractor = VideoPoseExtractor()

    combined_frames = []
    comparison_still = None
    decomposition_cells = []
    # 6 key frames covering all 5 physical phases:
    # 0 (F01: Diagonal approach), 6 (F07: Collar insertion), 11 (F12: Deep left slide),
    # 15 (F16: Partial withdrawal), 17 (F18: Left thumb slides down & needle drops!),
    # 25 (F26: Right index finger pointing on padlock body)
    decomp_indices = [0, 6, 11, 15, 17, 25]

    for i in range(num_frames):
        progress = i / max(1, num_frames - 1)
        v_frame = video_frames[i]
        h_vid, w_vid, _ = v_frame.shape

        human_meshes, state_info = extractor.reconstruct_bimanual_frame(
            progress=progress, use_robot_hands=False, video_path=clip_path
        )
        robot_meshes, _ = extractor.reconstruct_bimanual_frame(
            progress=progress, use_robot_hands=True, video_path=clip_path
        )

        human_cam = flip_y_for_camera(human_meshes)
        robot_cam = flip_y_for_camera(robot_meshes)

        tool_status = "Needle Active" if state_info["needle_visible"] else "Needle Dropped -> Index Pointing"

        img_twin = render_scene_3d(
            human_cam, elev=16, azim=-88,
            title="Stage 1: Connected 5-Finger Human Hand & Lock Twin",
            subtitle=f"{state_info['phase']} | {tool_status}",
            xlim=(-70, 78), ylim=(-55, 55), zlim=(-24, 86),
            width=w_vid, height=h_vid
        )

        img_robot = render_scene_3d(
            robot_cam, elev=16, azim=-88,
            title="Stage 2: Bimanual 5-Finger Dexterous Robot Hands",
            subtitle=f"Inspire / Unitree H1 Dexterous Hands | {tool_status} | t={progress*10.0:.1f}s",
            xlim=(-70, 78), ylim=(-55, 55), zlim=(-24, 86),
            width=w_vid, height=h_vid
        )

        v_annotated = v_frame.copy()
        cv2.putText(v_annotated, "Input: Monocular Video (30-Frame Ground Truth)", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (0, 0, 0), 4)
        cv2.putText(v_annotated, "Input: Monocular Video (30-Frame Ground Truth)", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (255, 255, 255), 2)
        cv2.putText(
            v_annotated,
            f"Frame {i+1:02d}/{num_frames} (t={progress*10.0:.1f}s) | {state_info['phase']}",
            (28, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 200), 2
        )

        target_h = 360
        target_w = int(w_vid * (target_h / h_vid))

        p1 = cv2.resize(v_annotated, (target_w, target_h))
        p2 = cv2.resize(img_twin, (target_w, target_h))
        p3 = cv2.resize(img_robot, (target_w, target_h))

        triple = np.hstack([p1, p2, p3])
        combined_frames.append(triple)

        if i == 11:
            comparison_still = triple.copy()

        if i in decomp_indices:
            pair_cell = np.hstack([p1, p2])
            decomposition_cells.append(pair_cell)

    os.makedirs(ASSETS_DIR, exist_ok=True)
    jpeg_path = os.path.join(ASSETS_DIR, "07_lpl_masterlock_end_to_end_comparison.jpg")
    cv2.imwrite(jpeg_path, cv2.cvtColor(comparison_still, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[+] Saved 3-panel comparison JPEG: {jpeg_path}")

    gif_path = os.path.join(ASSETS_DIR, "07_lpl_masterlock_end_to_end_animation.gif")
    imageio.mimsave(gif_path, combined_frames, fps=4, loop=0)
    print(f"[+] Saved slowed-down continuous GIF (30 frames @ 4 fps): {gif_path}")

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
