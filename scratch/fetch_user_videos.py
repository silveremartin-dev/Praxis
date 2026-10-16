"""
Téléchargement des 5 vidéos cibles soumises par l'utilisateur dans data/raw/.
Usage: py scratch/fetch_user_videos.py
"""

import os
import subprocess
import sys

TARGET_VIDEOS = [
    {
        "id": "RRunwAI7N5Q",
        "url": "https://www.youtube.com/watch?v=RRunwAI7N5Q",
        "filename": "01_LPL_DropKey_RRunwAI7N5Q",
        "title": "LockPickingLawyer - Drop Key Locks",
        "priority": "P0 (Court, macro fixe)"
    },
    {
        "id": "6K_jcXrtrx8",
        "url": "https://www.youtube.com/watch?v=6K_jcXrtrx8",
        "filename": "02_iFixit_DualSense_6K_jcXrtrx8",
        "title": "iFixit - PS5 DualSense Edge Teardown",
        "priority": "P0 (Court, zénithal atelier)"
    },
    {
        "id": "mA3QtWz7RMQ",
        "url": "https://www.youtube.com/watch?v=mA3QtWz7RMQ",
        "filename": "03_DIY_CementWD40_mA3QtWz7RMQ",
        "title": "DIY Bricolage - WD-40 & Ciment",
        "priority": "P1 (Moyen 15m)"
    },
    {
        "id": "KOBqgv-TpyA",
        "url": "https://www.youtube.com/watch?v=KOBqgv-TpyA",
        "filename": "04_Cuisine_Installation_KOBqgv-TpyA",
        "title": "Installation Cuisine en 1 semaine",
        "priority": "P2 (Long 35m, in situ)"
    },
    {
        "id": "f8NFSwb6VQ8",
        "url": "https://www.youtube.com/watch?v=f8NFSwb6VQ8",
        "filename": "05_Watch_Zodiac_f8NFSwb6VQ8",
        "title": "Wristwatch Revival - Zodiac Chronograph",
        "priority": "P2 (Long 58m, micro-mécanique)"
    }
]

def download_one(video_info, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, f"{video_info['filename']}.%(ext)s")
    
    # Format : max 720p pour rapidité et économie d'espace disque
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/best[height<=720]/best",
        "-o", out_template,
        "--no-playlist",
        video_info["url"]
    ]
    
    print(f"\n[*] Téléchargement de [{video_info['priority']}] : {video_info['title']}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"[+] Succès pour {video_info['filename']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] Erreur pour {video_info['filename']} : {e}")
        return False

if __name__ == "__main__":
    print(f"Démarrage du rapatriement des {len(TARGET_VIDEOS)} vidéos demandées...")
    for v in TARGET_VIDEOS:
        download_one(v)
    print("\nTous les téléchargements sont terminés.")
