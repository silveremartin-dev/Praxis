#!/usr/bin/env bash
# Provisionnement d'une VM Google Compute Engine avec pilote NVIDIA et environnement CUDA
set -euo pipefail

echo "Configuration de l'environnement GCP pour Pose / V2PDT..."

# Mise à jour système
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    ffmpeg \
    git \
    curl \
    python3-pip \
    python3-venv

# Installation des dépendances Python
python3 -m venv /opt/pose_venv
source /opt/pose_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Environnement prêt."
