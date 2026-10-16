#!/usr/bin/env bash
# Script d'exécution batch pour instance Google Cloud Platform (Compute Engine avec GPU L4/T4 ou Cloud Run Job)
set -euo pipefail

echo "=== Démarrage du pipeline de traitement batch sur GCP ==="

# 1. Vérification GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU détecté :"
    nvidia-smi
else
    echo "ATTENTION : Aucun GPU NVIDIA détecté. Le traitement 3D sera ralenti."
fi

# 2. Variables de configuration
GCS_BUCKET=${GCS_BUCKET:-"gs://pose-robot-dataset-raw"}
GCS_OUTPUT_BUCKET=${GCS_OUTPUT_BUCKET:-"gs://pose-robot-dataset-processed"}
WORK_DIR="/tmp/pose_processing"

mkdir -p "${WORK_DIR}/input" "${WORK_DIR}/output"

# 3. Synchronisation des vidéos brutes depuis Google Cloud Storage
echo "Synchronisation des vidéos brutes depuis ${GCS_BUCKET}..."
gsutil -m rsync -r "${GCS_BUCKET}" "${WORK_DIR}/input"

# 4. Exécution du pipeline
echo "Exécution du pipeline de transformation..."
python -m src.pipeline \
    --input_dir "${WORK_DIR}/input" \
    --output_dir "${WORK_DIR}/output" \
    --device "cuda" \
    --enable_sdf_sanitizer \
    --export_lerobot

# 5. Export des datasets validés vers GCS
echo "Sauvegarde des datasets traités vers ${GCS_OUTPUT_BUCKET}..."
gsutil -m rsync -r "${WORK_DIR}/output" "${GCS_OUTPUT_BUCKET}"

echo "=== Traitement batch terminé avec succès ==="
