🤖 Les robots humanoïdes manquent de démonstrations. YouTube en contient des millions d'heures.
Comment passer de l'une à l'autre sans "halluciner" la physique ?

C'est la question que j'explore avec Praxis, un projet open source (MIT) que je publie aujourd'hui.

Le problème : la téléopération (VR, ALOHA) ne passe pas à l'échelle, et entraîner directement sur les pixels de vidéos web pose trois soucis : des doigts qui traversent les objets en 3D monoculaire, aucune information de force, et l'écart entre une main humaine et une pince robotique.

L'approche : mettre l'outil et la pièce au centre (leur trajectoire est indépendante du robot), puis filtrer automatiquement tout ce qui n'est pas physiquement plausible. Un faible rendement est acceptable si le rejet se fait sans intervention humaine.

Ce que contient le dépôt :
🔹 découpage de vidéos en clips d'action + métadonnées JSON
🔹 contraintes de non-pénétration par SDF analytiques
🔹 dynamique inverse (masse, inertie, frottement de Coulomb)
🔹 audit de faisabilité IK (ALOHA ViperX, Unitree H1, Franka Panda ; seuil ≥ 85 %)
🔹 vérification de stabilité sous MuJoCo
🔹 export au format Hugging Face LeRobot (Parquet)
🔹 4 jumeaux numériques de démonstration (cadenas, horlogerie, machine Singer), 31 tests

Statut, en toute transparence : c'est un prototype de recherche. Les scènes 3D de démonstration sont des assemblages CAO paramétriques guidés par l'observation des vidéos, pas encore une reconstruction entièrement automatique. L'analyse des coûts est un modèle prospectif, pas une mesure sur un traitement en lot.

Retours, critiques et contributions bienvenus 👇
https://github.com/silveremartin-dev/Praxis

#Robotique #RobotsHumanoides #OpenSource #VisionParOrdinateur #DigitalTwin #MachineLearning #LeRobot
