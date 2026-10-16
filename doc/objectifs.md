# Objectifs du Projet : Video-to-Physical-Digital-Twin (V2PDT)

## 1. Vision et Proposition de Valeur
Le projet **Pose / Video-to-Physical-Digital-Twin (V2PDT)** vise à exploiter le gisement massif de vidéos Web (YouTube) pour extraire automatiquement des démonstrations de manipulation dextre et les transformer en **jumeaux numériques physiques exploitables en simulation robotique**.

Plutôt que d'entraîner des modèles visuomoteurs directement sur des pixels bruités et sans calibrage physique, le pipeline extrait :
1. Les trajectoires cinématiques 3D de la main humaine (modèle anatomique MANO).
2. L'identification et la pose 6D d'objets rigides standardisés (outils, pièces, serrures, vis).
3. L'optimisation conjointe garantissant **l'absence stricte d'interpénétration** et la **stabilité des contacts**.
4. L'estimation des grandeurs dynamiques (masse, frottement, inertie) par dynamique inverse visuelle.
5. La certification de faisabilité mécanique pour des robots cibles (via modèles URDF).

---

## 2. Objectifs Clés (KPIs & Métriques de Réussite)

| Objectif | Cible / Métrique | Description |
| :--- | :--- | :--- |
| **Précision géométrique de contact** | Pénétration < 1 mm | Élimination totale des artefacts où les doigts traversent l'objet. |
| **Validation physique automatique** | Taux d'échec silencieux < 2 % | Certification dans MuJoCo/Isaac Sim sans instabilité numérique. |
| **Standardisation des données** | Format LeRobot / Parquet | Compatibilité directe avec l'écosystème Hugging Face LeRobot. |
| **Indépendance vis-à-vis du robot** | Trajectoire d'outil agnostique | Données de manipulation exploitables par tout bras ou humanoïde. |
| **Filtrage de faisabilité cinématique** | Score IK 0-100 % | Tri automatisé des gestes compatibles avec les limites d'articulations d'un URDF. |

---

## 3. Périmètre d'Application Initial (Operational Design Domain)

Pour éviter les pièges d'effondrement de modèle et de simulation complexe (matière déformable, fluides) :
* **Objets inclus :** Solides rigides standardisés (outils à main, tournevis, clés, cadenas, composants électroniques, visserie, boutons/rotacteurs).
* **Objets exclus (V1) :** Matières molles (pâte, tissus), liquides, découpe destructive (sciage, hachage).
* **Sources cibles :** Vidéos avec caméras macro/zénithales stables, éclairage studio uniforme, mouvements délibérés (ex: LockPickingLawyer, iFixit, horlogerie, montage PC).
