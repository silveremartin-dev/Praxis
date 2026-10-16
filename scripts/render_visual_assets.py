"""
Visual Asset Generator for Praxis:
Generates high-resolution side-by-side comparison JPEGs (RGB video frame vs. 3D digital twin)
and smooth 360-degree rotating animated GIFs for all processed demonstrations.
"""

import os
import shutil
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh
import imageio

ARTIFACT_DIR = r"C:\Users\silve\.gemini\antigravity\brain\5eb7cf8d-a6d8-47a0-ad1d-925676cd6bf3"

SCENES_CONFIG = [
    {
        "id": "01_lpl_dropkey",
        "title": "LockPickingLawyer - Drop Key Precision Demonstration",
        "clip_mp4": "data/clips/01_LPL_DropKey_RRunwAI7N5Q_clip_18s_26s.mp4",
        "obj_path": "data/processed/01_lpl_dropkey_scene_3d.obj",
        "elev": 25,
        "azim": -60,
        "face_color": "#4da6ff",
        "frame_sec": 4.0
    },
    {
        "id": "05_watch_zodiac",
        "title": "Wristwatch Revival - Chronograph Gear Micro-Extraction",
        "clip_mp4": "data/clips/05_Watch_Zodiac_f8NFSwb6VQ8_clip_595s_608s.mp4",
        "obj_path": "data/processed/05_watch_zodiac_scene_3d.obj",
        "elev": 35,
        "azim": -45,
        "face_color": "#e6b800",
        "frame_sec": 6.5
    },
    {
        "id": "06_singer_sewingmachine",
        "title": "1894 Singer Sewing Machine - Cast Iron Faceplate Detachment",
        "clip_mp4": "data/clips/06_Singer_SewingMachine_BMNgIfLHfjk_clip_118s_129s.mp4",
        "obj_path": "data/processed/06_singer_sewingmachine_scene_3d.obj",
        "elev": 20,
        "azim": -70,
        "face_color": "#708090",
        "frame_sec": 5.0
    }
]

def extract_video_frame(clip_path: str, at_second: float) -> np.ndarray:
    """Extracts a single RGB frame from a video at the specified timestamp."""
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame = int(at_second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        # Fallback dummy frame
        return np.full((720, 1280, 3), 40, dtype=np.uint8)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def render_mesh_to_image(mesh: trimesh.Trimesh, elev: float, azim: float, title: str, face_color: str, width: int = 800, height: int = 600) -> np.ndarray:
    """Renders a 3D mesh to an RGB image array with dark robotic studio style."""
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111318')
    fig.patch.set_facecolor('#111318')
    
    vertices = mesh.vertices
    faces = mesh.faces
    # Decimate faces if too dense for fast rendering
    step = max(1, len(faces) // 600)
    sampled_faces = faces[::step]
    
    poly3d = Poly3DCollection(vertices[sampled_faces], alpha=0.9, edgecolor='#222831', linewidths=0.25)
    poly3d.set_facecolor(face_color)
    ax.add_collection3d(poly3d)
    
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color='#00ffcc', fontsize=12, pad=15, weight='bold')
    ax.tick_params(colors='#555555', labelsize=8)
    ax.grid(color='#252830', linestyle=':')
    
    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = rgba[:, :, :3].copy()
    plt.close(fig)
    return img

def create_comparison_jpeg(scene_cfg: dict, out_dir: str = "data/processed") -> str:
    """Creates a side-by-side comparison JPEG of original video frame vs 3D digital twin."""
    rgb_frame = extract_video_frame(scene_cfg["clip_mp4"], scene_cfg["frame_sec"])
    mesh = trimesh.load(scene_cfg["obj_path"])
    
    h, w, _ = rgb_frame.shape
    mesh_render = render_mesh_to_image(
        mesh,
        elev=scene_cfg["elev"],
        azim=scene_cfg["azim"],
        title="Praxis 3D Physical Digital Twin (SDF Certified)",
        face_color=scene_cfg["face_color"],
        width=w,
        height=h
    )
    
    # Resize mesh render to match video frame height
    if mesh_render.shape[:2] != (h, w):
        mesh_render = cv2.resize(mesh_render, (w, h))
        
    # Annotate video frame
    annotated_rgb = rgb_frame.copy()
    cv2.putText(annotated_rgb, "Monocular In-The-Wild Video", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(annotated_rgb, "Monocular In-The-Wild Video", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(annotated_rgb, scene_cfg["title"], (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
    
    # Combined side-by-side
    combined = np.hstack([annotated_rgb, mesh_render])
    
    out_path = os.path.join(out_dir, f"{scene_cfg['id']}_comparison.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[+] Comparison JPEG generated: {out_path}")
    
    # Copy to brain artifact directory
    if os.path.exists(ARTIFACT_DIR):
        artifact_dest = os.path.join(ARTIFACT_DIR, f"{scene_cfg['id']}_comparison.jpg")
        shutil.copy2(out_path, artifact_dest)
        
    return out_path

def create_rotating_gif(scene_cfg: dict, out_dir: str = "data/processed", num_frames: int = 24) -> str:
    """Generates a smooth 360-degree rotating animation GIF of the reconstructed 3D digital twin."""
    mesh = trimesh.load(scene_cfg["obj_path"])
    frames = []
    
    azimuths = np.linspace(0, 360, num_frames, endpoint=False)
    for azim in azimuths:
        frame_img = render_mesh_to_image(
            mesh,
            elev=scene_cfg["elev"],
            azim=azim,
            title=f"3D Twin Orbit: {scene_cfg['id']}",
            face_color=scene_cfg["face_color"],
            width=500,
            height=380
        )
        frames.append(frame_img)
        
    out_path = os.path.join(out_dir, f"{scene_cfg['id']}_animation.gif")
    imageio.mimsave(out_path, frames, fps=12, loop=0)
    print(f"[+] 360 Rotation GIF generated: {out_path}")
    
    if os.path.exists(ARTIFACT_DIR):
        artifact_dest = os.path.join(ARTIFACT_DIR, f"{scene_cfg['id']}_animation.gif")
        shutil.copy2(out_path, artifact_dest)
        
    return out_path

def main():
    os.makedirs("data/processed", exist_ok=True)
    if os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        
    for cfg in SCENES_CONFIG:
        print(f"\n--- Generating visual assets for: {cfg['id']} ---")
        create_comparison_jpeg(cfg)
        create_rotating_gif(cfg)
        
    print("\n[OK] All visual assets successfully generated.")

if __name__ == "__main__":
    main()
