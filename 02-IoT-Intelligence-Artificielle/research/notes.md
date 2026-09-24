# Notes de recherche

## Choix du dataset

Trois options envisagées : UCI HAR, WISDM, MotionSense (voir comparatif dans le guide de travail).
Choix final : **UCI HAR**, pour trois raisons :
- split train/test déjà fait par sujet par les auteurs (fiable, documenté),
- signaux bruts disponibles (`Inertial Signals/`) en plus des 561 features précalculées,
- très largement utilisé et documenté, donc peu de risque de mauvaise interprétation du format.

Argument retenu pour la vidéo : on utilise les signaux **bruts**, pas les features précalculées,
parce qu'un vrai ESP32 ne recevrait jamais de features déjà calculées, seulement des valeurs brutes
du capteur.

Source : page officielle du dataset sur UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), et
l'article original : Anguita, D., Ghio, A., Oneto, L., Parra, X., Reyes-Ortiz, J.L. (2013).
*A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN.

## Expérience : modèle plus petit (`experiment_smaller_cnn.py`)

**Question :** le CNN final (16/32/32 filtres, 7686 paramètres) est-il plus gros que nécessaire ?

**Changement testé :** mêmes couches, filtres divisés par deux (8/16/16), couche dense 16 au lieu
de 32 → 2182 paramètres (3.5x moins).

**Résultat :** meilleure accuracy de validation que le modèle final (98.45 % contre 97.33 %),
en un peu plus d'époques avant l'arrêt anticipé (60 contre 27).

**Décision : modèle final conservé tel quel.** Cette variante plus petite n'a été évaluée que sur
la validation, jamais sur le test : l'utiliser aurait demandé de refaire toute la chaîne déjà figée
(évaluation test avec matrice de confusion, export TFLite float32/int8, vérification, script de
simulation, benchmark) avec la deadline qui approchait. C'est une **piste d'amélioration réelle**,
mentionnée aussi dans le README, plutôt qu'un changement fait à la dernière minute sans avoir eu le
temps de la valider correctement de bout en bout.

**Ce que ça confirme :** la validation seule ne suffit pas à comparer deux architectures de façon
fiable, c'est justement pour ça que l'évaluation finale se fait uniquement sur le test, une fois le
modèle choisi et figé (voir `04_evaluation_export.ipynb`).
