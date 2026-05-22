# HQDR-Net: Hybrid Quantum Data Re-uploading Network for UAV Wildfire Detection

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-QML-purple.svg)](https://pennylane.ai/)
[![Docker](https://img.shields.io/badge/Docker-Edge%20Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

**HQDR-Net** is a production-ready, edge-deployable **Hybrid Quantum-Classical deep learning framework** for real-time wildfire detection from UAV (Unmanned Aerial Vehicle) aerial imagery.

The architecture fuses a **classical convolutional feature extractor** (MobileNetV3-Small) with an **8-qubit Variational Quantum Circuit (VQC)** built on the **Data Re-uploading** strategy. This allows the model to capture complex, non-linear feature correlations in a simulated quantum Hilbert space — while executing in **~85ms per frame on a standard edge CPU** with a total footprint of **~18.6 MB**.

This repository contains:
- **Hybrid-QPCA Model** — The proposed quantum-classical model.
- **Baseline Models** — Classical-PCA, EfficientNet-B0, MobileNetV2, and ResNet-18 for comparison.
- **Edge Deployment Pipeline** — A full Dockerized REST API for offline UAV inference.
- **Training Notebooks** — End-to-end training scripts for Kaggle/Colab.

---

## Architecture

```
Input Frame (224x224)
        │
        ▼
┌─────────────────────────┐
│  MobileNetV3-Small      │  ← Classical Feature Extractor (ImageNet pretrained)
│  Backbone               │  → Outputs 576-dim feature vector
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Pre-Projection Layer   │  ← Linear: 576 → 16 (8 qubits × 2 angle params)
│  (Angle Encoder)        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  8-Qubit Variational    │  ← Ry(θ) rotations per qubit
│  Quantum Circuit (VQC)  │  ← Circular CNOT entanglement ring
│  Data Re-Uploading      │  ← Repeated encoding layers (L=2)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  HSV Heuristic Filter   │  ← Fire Pixel Ratio (FPR) calibration
│  (Dual Verification)    │  ← Suppresses sunset / canopy false positives
└───────────┬─────────────┘
            │
            ▼
  Binary Classification
  (Fire / No Fire)
```

---

## Model Performance (UAVS-FDDB Dataset)

Evaluated on a balanced test set of **17,203 images** across 4 diurnal scenarios (Evening Fire, Pre-Evening Fire, Evening Forest, Pre-Evening Forest).

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Size | CPU Latency |
|:------|:--------:|:---------:|:------:|:--------:|:--------:|:----:|:-----------:|
| **Hybrid-QPCA (Proposed)** | **99.94%** | **100.00%** | **100.00%** | **100.00%** | **1.000** | **18.6 MB** | **~85 ms** |
| Classical-PCA Baseline | 99.11% | 99.10% | 98.70% | 98.90% | 0.999 | 17.1 MB | ~78 ms |
| EfficientNet-B0 Full | 99.88% | 99.89% | 99.87% | 99.88% | 1.000 | 17.7 MB | ~142 ms |
| MobileNetV2 | 99.65% | 99.63% | 99.59% | 99.61% | 0.999 | 9.1 MB | ~110 ms |
| ResNet-18 | 99.53% | 99.50% | 99.40% | 99.45% | 0.999 | 44.8 MB | ~195 ms |

> **Key result:** HQDR-Net achieves **100% Recall** (zero missed fire events) on the full test set while maintaining the lowest practical CPU-edge latency among high-accuracy models.

---

## Repository Structure

```
HQDR_Net_GitHub_Release/
│
├── README.md                      ← This file
├── requirements.txt               ← Python dependencies
├── LICENSE
│
├── models/
│   ├── Hybrid-QPCA_Proposed_best.pt   ← PROPOSED: Hybrid Quantum-Classical model
│   ├── Classical-PCA_best.pt          ← Baseline: Classical PCA variant
│   ├── EfficientNet-B0_Full_best.pt   ← Baseline: EfficientNet-B0
│   ├── MobileNetV2_best.pt            ← Baseline: MobileNetV2
│   └── ResNet-18_best.pt              ← Baseline: ResNet-18
│
├── edge_deployment/
│   ├── Dockerfile                     ← Docker build file (python:3.10-slim)
│   ├── edge_tracker.py                ← Main inference engine (fire alert extractor)
│   ├── step1_onnx_quantizer.py        ← Converts .pt model to ONNX format
│   ├── step3_pi.py                    ← Lightweight Raspberry Pi inference script
│   ├── setup_and_run.sh               ← One-shot setup script for Linux/Pi
│   └── test_onnx.py                   ← ONNX model validation script
│
└── notebooks/
    └── HQDR_Net_Training_Notebook.ipynb   ← End-to-end training notebook (Kaggle/Colab)
```

---

## Quickstart: Edge Deployment (Docker)

### Prerequisites
- Docker Desktop (Windows) or Docker Engine (Linux/Raspberry Pi)
- Your UAV video clips placed in `edge_deployment/clips/`

### Step 1: Build the Docker Image
```bash
cd edge_deployment
docker build -t hqdr-net-edge .
```

### Step 2: Run Fire Detection
```bash
docker run --rm \
  -v $(pwd)/clips:/app/clips \
  -v $(pwd)/Fire_Alerts:/app/Fire_Alerts \
  hqdr-net-edge
```

Fire alert images will be saved to the `Fire_Alerts/` folder with:
- Detection confidence overlaid on the frame
- Red bounding border around confirmed fire frames
- Filename encoding the frame index and confidence score

---

## Quickstart: Direct Python Inference

```bash
pip install -r requirements.txt
cd edge_deployment
python edge_tracker.py
```

---

## Training

Open the notebook `notebooks/HQDR_Net_Training_Notebook.ipynb` in Kaggle or Google Colab.

**Dataset:** [UAVS-FDDB (UAV-based Forest Fire Detection Database)](https://www.kaggle.com/datasets/)

The notebook covers:
1. Dataset download and preprocessing (augmentation pipeline)
2. MobileNetV3-Small backbone extraction
3. Variational Quantum Circuit (VQC) construction using PennyLane
4. Hybrid end-to-end training loop with early stopping
5. Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix
6. Model export to `.pt` and `.onnx`

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
pennylane>=0.35.0
opencv-python>=4.8.0
Pillow>=9.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
onnx>=1.14.0
onnxruntime>=1.16.0
```

---

## Dataset

**UAVS-FDDB** (UAV-based Forest Fire Detection Database)  
- 17,203 labelled aerial images  
- 4 Scenarios: Evening Fire, Pre-Evening Fire, Evening Forest, Pre-Evening Forest  
- Balanced binary classification (Fire / No Fire)  
- Available on [Kaggle](https://www.kaggle.com/)

---

## Citation

If you use this work, please cite:

```bibtex
@article{jayadharun2026hqdrnet,
  title   = {HQDR-Net: A Hybrid Quantum Data Re-uploading Neural Network for Real-Time UAV Wildfire Detection},
  author  = {Jaya Dharun R},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  note    = {VIT University, Vellore}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Remote ProOps Engineering Division**  
remoteproops@gmail.com
