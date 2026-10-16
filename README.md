# Pose : Video-to-Physical-Digital-Twin (V2PDT)

Système automatisé de transformation de vidéos Web monoculaires (YouTube) en démonstrations physiques 3D sans collision pour l'entraînement de robots humanoïdes.

---

## 📌 Architecture des Répertoires

```
Pose/
├── data/
│   ├── raw/              # Vidéos YouTube brutes téléchargées (mp4)
│   ├── clips/            # Segments d'action découpés (5-15s)
│   └── processed/        # Trajectoires 3D et datasets prêts pour LeRobot
├── doc/
│   ├── agents.md         # Rôles, prompts et responsabilités des agents IA
│   ├── objectifs.md      # Objectifs, métriques, et périmètre opérationnel (ODD)
│   ├── roadmap.md        # Feuille de route en 5 phases (de local à GCP)
│   └── architecture.md   # Spécifications d'architecture système et flux
├── scratch/              # Scripts d'exploration, de test et de prototypage rapide
├── scripts/
│   └── gcp/              # Scripts de batch processing et déploiement sur Google Cloud
└── src/
    ├── ingestion/        # Téléchargement et découpage temporel d'actions (VLM)
    ├── pose_extraction/  # Extraction des maillages 3D main (HaMeR) et outil (SAM 2)
    ├── physics_sanitizer/# Optimisation SDF non-pénétration et contraintes de contact
    ├── feasibility/      # Solveur IK et filtre de faisabilité sur robot cible (URDF)
    ├── exporter/         # Formatage normalisé LeRobot / Parquet & Rerun.io
    └── pipeline.py       # Orchestrateur CLI principal
```

---

## 🚀 Démarrage Rapide

### 1. Installation des dépendances
```bash
py -m pip install -r requirements.txt
```

### 2. Téléchargement d'échantillons vidéo
```bash
py scratch/download_samples.py
```

### 3. Exécution du pipeline
```bash
py -m src.pipeline --input_dir data/raw --output_dir data/processed
```

---

## 📖 Documentation Complète
* Consultez [doc/agents.md](file:///c:/Silvere/Encours/Developpement/Pose/doc/agents.md) pour les rôles des agents.
* Consultez [doc/objectifs.md](file:///c:/Silvere/Encours/Developpement/Pose/doc/objectifs.md) pour les objectifs et critères de succès.
* Consultez [doc/roadmap.md](file:///c:/Silvere/Encours/Developpement/Pose/doc/roadmap.md) pour la feuille de route.
* Consultez [doc/architecture.md](file:///c:/Silvere/Encours/Developpement/Pose/doc/architecture.md) pour les détails techniques.
