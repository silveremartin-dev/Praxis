# Agents IA & Rôles dans le Système

Ce document décrit les responsabilités, prompts système, entrées et sorties des agents autonomes et modules d'IA composant l'architecture Video-to-Physical-Digital-Twin.

---

## 1. Cartographie des Agents

| Nom de l'Agent | Type | Responsabilité Principale |
| :--- | :--- | :--- |
| **Ingestion & Slicer Agent** | VLM (Gemini / Qwen) | Analyse sémantique de la vidéo et découpage en segments d'action continus. |
| **Pose & Mesh Recovery Agent** | Modèle de vision local (HaMeR) | Extraction des maillages MANO de la main et suivi 3D des articulations. |
| **Tool & CAD Matcher Agent** | SAM 2 + Retrieval DINOv2 | Détection 2D/3D de l'outil et sélection du modèle CAO canonique correspondant. |
| **Contact Sanitizer Agent** | Optimiseur géométrique (SDF) | Ajustement sans interpénétration des maillages main/outil et attraction de contact. |
| **Physical Dynamics Agent** | Solveur de dynamique inverse | Estimation de la masse, du centre de gravité et des forces appliquées. |
| **Robot Feasibility Auditor** | Solveur IK (Pinocchio/PyBullet) | Calcul de la compatibilité cinématique avec le robot cible et arbitrage rejet/validation. |
| **Dataset Packager Agent** | Script d'export automatisé | Formatage, sérialisation Parquet et validation du schéma LeRobot. |

---

## 2. Spécification Détaillée des Agents

### Agent 1 : Ingestion & Slicer Agent
* **Rôle :** Inspecteur vidéo de haut niveau.
* **Modèle :** Gemini 1.5 Flash (via API) ou Qwen2.5-VL en local.
* **Prompt Type :**
  > *"Tu es un expert en analyse de manipulation robotique. Examine cette vidéo échantillonnée à 1 image par seconde. Identifie les séquences continues (sans coupure de caméra, caméra stable) où une main humaine interagit physiquement avec un objet rigide (outil, pièce, bouton, serrure). Pour chaque séquence, fournis : start_time, end_time, action_verb, tool_name, workpiece_name, camera_stability_score (0 à 1)."*
* **Sortie attendue :** Format JSON strict validé par Pydantic.

### Agent 2 : Pose & Mesh Recovery Agent
* **Rôle :** Reconstructeur anatomique de la main.
* **Technologie :** HaMeR (Hand Mesh Recovery) avec modèle MANO.
* **Entrée :** Frames RGB découpées du clip d'action.
* **Sortie :** Matrice de rotation globale du poignet, 21 coordonnées d'articulations 3D (`X, Y, Z`), et maillage de 778 sommets pour chaque frame.

### Agent 3 : Tool & CAD Matcher Agent
* **Rôle :** Identificateur d'objets et géomètre.
* **Technologie :** SAM 2 (Segment Anything 2) pour le masque de propagation + DINOv2 pour l'appariement avec la banque CAO Objaverse.
* **Sortie :** Modèle 3D triangulé (`.obj` / `.stl`) avec pose 6D estimée ($SE(3)$ : position relative et orientation quaternion).

### Agent 4 : Contact Sanitizer Agent
* **Rôle :** Arbitre physique des collisions.
* **Technologie :** Optimisation différentiable PyTorch basée sur les champs de distance signée (SDF).
* **Règle fondamentale :** 
  * Aucun sommet de la main ne peut avoir une distance signée négative par rapport au maillage de l'outil ($\text{SDF}(v) \ge 0$).
  * En phase de préhension, la distance minimale entre au moins deux doigts et la poignée de l'outil doit être inférieure à un seuil $\epsilon = 1.0 \text{ mm}$.

### Agent 5 : Robot Feasibility Auditor
* **Rôle :** Gardien de la transférabilité sur robot réel.
* **Technologie :** Cinématique inverse (IK) basée sur le fichier URDF du robot cible (ex: ALOHA, Unitree H1, Fourier GR-1).
* **Critères d'évaluation :**
  * Toutes les positions cibles sont-elles atteignables sans dépasser les limites angulaires des moteurs ?
  * La vitesse requise dépasse-t-elle le couple ou la vitesse maximale des actionneurs ?
* **Action :** Si le score est $< 85\%$, la séquence est étiquetée comme non transférable et écartée du dataset robotique.
