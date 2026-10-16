# Roadmap de Développement : Video-to-Physical-Digital-Twin

Cette feuille de route détaille les 5 phases de mise en œuvre, du prototype local sur PC jusqu'au passage à l'échelle sur Google Cloud Platform (GCP).

---

## Phase 1 : Socle, Ingestion & Données Pilotes (Semaine 1)
- [x] Initialisation du dépôt Git, arborescence (`data/`, `doc/`, `scratch/`, `scripts/gcp/`, `src/`).
- [x] Spécification complète (`doc/objectifs.md`, `doc/architecture.md`, `doc/agents.md`, `doc/roadmap.md`).
- [ ] Téléchargement d'un lot pilote de vidéos représentatives dans `data/raw/` (LockPickingLawyer, iFixit).
- [ ] Création du script de découpage sémantique assisté par VLM (`src/ingestion/slicer.py`).
- [ ] Validation du pipeline de découpage en segments d'actions de 5 à 15 secondes (`data/clips/`).

---

## Phase 2 : Extraction Cinématique & Visualisation (Semaine 2)
- [ ] Intégration de l'extraction de pose de main (HaMeR / MANO) dans `src/pose_extraction/`.
- [ ] Intégration du tracking d'outil et masque 2D/3D (SAM 2).
- [ ] Mise en place du visualiseur synchronisé avec **Rerun.io** pour inspecter en temps réel la vidéo à gauche et le squelette 3D à droite.
- [ ] Calibration de l'échelle métrique monoculaire par rapport aux dimensions standardisées des outils/objets.

---

## Phase 3 : Assainissement Physique des Contacts (Semaine 3)
- [ ] Développement du module d'optimisation sans interpénétration (`src/physics_sanitizer/sdf_optimizer.py`).
- [ ] Modélisation de la contrainte de contact et force-closure sur les phases de manipulation.
- [ ] Benchmark "Avant / Après" assainissement sur les trajectoires de préhension d'un outil rigide.
- [ ] Export des trajectoires assainies au format standard **LeRobot** (Parquet / Arrow).

---

## Phase 4 : Filtre de Faisabilité Robot & Validation MuJoCo (Semaine 4)
- [ ] Intégration des fichiers URDF de référence (ex: bras ALOHA / pince 2 doigts et main anthropomorphe type Unitree/Fourier).
- [ ] Solveur cinématique inverse (IK) pour calculer le score de faisabilité mécanique (`src/feasibility/urdf_filter.py`).
- [ ] Injection automatique de la scène virtuelle et de la trajectoire dans un environnement de test **MuJoCo**.
- [ ] Établissement du filtre de rejet automatique (rejet sans intervention humaine si instabilité physique ou singularité).

---

## Phase 5 : Industrialisation & Scaling Cloud (Google Cloud Platform)
- [ ] Conteneurisation Docker complète avec support GPU NVIDIA (CUDA).
- [ ] Scripts de traitement batch sur **Google Cloud Run** ou instances **GCE avec GPU (L4 / T4)** (`scripts/gcp/`).
- [ ] Automatisation du pipeline d'ingestion massive via des playlists YouTube ciblées.
- [ ] Publication de la première version du dataset open-source validé sur le hub Hugging Face.
