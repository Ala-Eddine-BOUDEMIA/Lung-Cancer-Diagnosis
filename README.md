# Lung Cancer Diagnosis
 
## Articles : 

1. "Pathologist-level classification of histologic patterns on resected lung adenocarcinoma slides with deep neural networks."

2. "Convolutional neural networks can accurately distinguish four histologic growth patterns of lung adenocarcinoma in digital slides"

3. "A Multi-resolution Deep Learning Framework for Lung Adenocarcinoma Growth Pattern Classification: 22nd Conference, MIUA 2018, Southampton, UK, July 9-11, 2018, Proceedings."

## Approche :

### Pre-processing :

- Utilisation des 26 WSI annotées de l'article (2) pour générer des patches pour 5 classes.

- Les classes sont :  Acinar, Solid, Micropapillary, Cribriform, Non-Tumor

- Augmentation du nombre des patches par des techniques de traitement d'images.

### 1-Split :

- Distribution aléatoire et équilibrée des patches générés sur les 5 classes.

- Nombre de patches souhaité par classe est de 10,000 à 20,000.

### 2-Train-Validate-Test :

- Entrainer, valider et tester le model (resnet18).

- Utiliser le model pour annoté le reste de la base de données (LUAD, Lung and Colon Cancer Histopathological Image Dataset (LC25000))

- Redistribuer les patches.

- Entrainer, valider et tester un nouveau model (resnet ou inception).

### 3-Evaluation :

- Aggrégation de prédictions des patches par l'une des méthodes suivantes :

    1. Majority voting.
    2. Patch averages.
    3. Thresholding.
    4. Attention-Based Classification. 

- Calculer les matrices : exactitude, précision, et score F1 par classe. 

- Visualisation des résultats (segmentation nécessaire).