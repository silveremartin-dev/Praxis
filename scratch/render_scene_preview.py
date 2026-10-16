"""
Rendu visuel 3D de la scène reconstruite avec matplotlib et trimesh.
Génère une image PNG haute résolution pour inspection visuelle immédiate.
"""

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh

def render_obj_preview(obj_path, output_png):
    mesh = trimesh.load(obj_path)
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    
    # Échantillonnage de faces pour un affichage rapide et propre
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Poly3DCollection
    poly3d = Poly3DCollection(vertices[faces[::2]], alpha=0.85, edgecolor='#333333', linewidths=0.3)
    poly3d.set_facecolor('#4da6ff') # Bleu tech robotique
    ax.add_collection3d(poly3d)
    
    # Bornes
    scale = np.concatenate([vertices.min(axis=0), vertices.max(axis=0)])
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    
    ax.view_init(elev=28, azim=-55)
    ax.set_title("Pose / V2PDT : Reconstruction 3D (LPL Drop Key Scene)", color='white', fontsize=14, pad=20)
    ax.tick_params(colors='gray')
    ax.grid(color='#333333', linestyle=':')
    
    plt.tight_layout()
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Aperçu 3D rendu avec succès : {output_png}")

if __name__ == "__main__":
    render_obj_preview("data/processed/01_lpl_dropkey_scene_3d.obj", "data/processed/01_lpl_dropkey_render_3d.png")
