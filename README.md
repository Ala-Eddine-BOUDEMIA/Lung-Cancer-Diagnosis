# DeepLUAD: A Deep Learning Pipeline for Lung Cancer Diagnosis

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange.svg)](https://pytorch.org/)

This repository contains the code for my final-year engineering thesis project. It presents a complete, end-to-end pipeline for classifying histopathological growth patterns in lung adenocarcinoma from Whole Slide Images (WSIs) and for classifying cancer types in other histopathology datasets.

---

## Project Overview & Key Achievements

This project was designed as a practical and efficient approach to clinically important problems in digital pathology, emphasizing a lightweight architecture and a reproducible workflow.

*   **High-Impact Application:** Addresses the automated classification of LUAD growth patterns, a critical but subjective task for pathologists.
*   **Strong Performance:** Achieved **96% slide-level accuracy** in classifying LUAD growth patterns on a key public dataset, outperforming reported pathologist benchmarks (~80-85%).
*   **Efficient Architecture:** Successfully implemented a fine-tuned **ResNet-18 model**, demonstrating that a computationally modest architecture can achieve strong performance.
*   **End-to-End Pipeline:** Developed the entire workflow, from WSI preprocessing (tiling, augmentation) and model training to a practical WSI-level aggregation of patch probabilities.
*   **Scientific Documentation:** The complete methodology and results are detailed in a **[preprint on bioRxiv](https://www.biorxiv.org/content/10.1101/2022.05.06.490977v1)**.

---

## Pipeline & Usage

This repository contains the code for three related tasks, structured into separate folders. The core pipeline consists of sequential scripts for preprocessing, data augmentation, training, and evaluation.

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Ala-Eddine-BOUDEMIA/Lung-Cancer-Diagnosis.git

# Install dependencies
cd Lung-Cancer-Diagnosis
pip install -r requirements.txt
```

### 2. Running a Task (Example: Growth Pattern Classification)
The workflow is designed to be run from within the appropriate task folder (e.g., `1-Growth_patterns_classification/`).

```bash
# 1. Preprocess raw WSIs into trainable patches
python3 1-Preprocessing.py --Overlap 0.45

# 2. Augment data to balance classes
python3 2-Processing.py --Maximum 10000

# 3. Split data into train/validation/test sets
python3 3-split.py --Validation_Set_Size 1000 --Test_Set_Size 1000

# 4. Train the model
python3 4-Train_val.py --num_epochs 200 --batch_size 32

# 5. Evaluate the model and aggregate results
python3 5-test.py
python3 6-Evaluation.py
```
*(Note: Care was taken to ensure strict slide-level separation between training, validation, and test sets to prevent data leakage.)*

### 3. Monitoring
Training can be monitored using TensorBoard:
```bash
tensorboard --logdir=Tensorboard
```

---

## Methodological Notes & Known Limitations

*   **Slide-Level Aggregation:** The pipeline's WSI-level prediction aggregates patch probabilities to identify the single *predominant* pattern. It was not designed to detect or quantify minor or secondary growth patterns within a single slide.

*   **Evaluation Metrics:** Performance is measured using standard classification metrics (Accuracy, Precision, Recall). The current implementation does not include metrics that account for inter-observer agreement, such as Cohen's kappa.

*   **Code Optimization:** The Python scripts are functional proofs-of-concept and are not optimized for large-scale performance (e.g., via multiprocessing).

*   **Stain Normalization:** The pipeline does not include a module for stain normalization to account for variations that may arise from different medical centers or scanning protocols.

---

## Sources & Citations
This work was made possible by public datasets and built upon the insights from the following key papers:

*   Wei, J., et al. "Pathologist-level Classification of Histologic Patterns on Resected Lung Adenocarcinoma Slides with Deep Neural Networks", *Scientific Reports* (2019).
*   Gertych, A., et al. "Convolutional neural networks can accurately distinguish four histologic growth patterns of lung adenocarcinoma in digital slides", *Scientific Reports* (2019).

