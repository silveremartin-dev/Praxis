"""
Script d'exploration et de téléchargement de vidéos échantillons dans data/raw/.
Usage: py scratch/download_samples.py
"""

import os
import subprocess
import sys

SAMPLE_VIDEOS = [
    {
        "name": "LPL_AmericanLock_1100",
        "url": "https://www.youtube.com/watch?v=StZCIDBKI-A",
        "category": "padlock_picking_gutting",
        "description": "LockPickingLawyer - American Lock 1100 Picked and Gutted (macro fixe, goupilles & clés)"
    },
    {
        "name": "LPL_MasterLock_No3",
        "url": "https://www.youtube.com/watch?v=uOO8yA2eU7M",
        "category": "padlock_bypass",
        "description": "LockPickingLawyer - Master Lock No. 3 (clés, manipulation directe et rotation)"
    }
]

def download_video(url: str, output_name: str, output_dir: str = "data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, f"{output_name}.%(ext)s")
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[height<=720][ext=mp4]/bestvideo[ext=mp4]/bestvideo/best",
        "-o", out_template,
        "--no-playlist",
        url
    ]
    
    print(f"[*] Téléchargement de {output_name}...")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"[+] Succès pour {output_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] Erreur lors du téléchargement de {output_name} : {e.stderr}")
        return False

if __name__ == "__main__":
    print(f"Démarrage du téléchargement des vidéos de référence ({len(SAMPLE_VIDEOS)} vidéos)...")
    for item in SAMPLE_VIDEOS:
        download_video(item["url"], item["name"])
    print("Téléchargements terminés.")
