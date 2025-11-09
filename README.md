# Official repository for the "Interpretable Arrhythmia Detection in ECG Scans Using Deep Learning Ensembles: A Genetic Programming Approach" paper.

## Overview

This repository provides tools for training, inference, and explainability (XAI) of ECG-based arrhythmia detection models.  
It focuses on making deep learning models both accurate and interpretable by combining several explainability techniques into one unified approach.

## How the Inferece Works

The models are trained to detect arrhythmias from ECG scans using a deep learning ensemble.  
Each model in the ensemble learns to recognize specific heartbeat patterns.  
A genetic programming structure called the **Giraffe Tree** combines these models to improve accuracy and generalization.  
Training and inference are implemented using **PyTorch Lightning**, ensuring organized workflows and reproducible results.

## How XAI Works

The explainability system helps visualize why the model made a certain prediction.  
It combines several attribution methods such as Saliency, Occlusion, Integrated Gradients, GradientShap, and LIME into a single explanation using **Ensemble Explainability**.  

Each of these methods highlights different aspects of the model’s decision.  
By merging them together, the system produces a more stable and reliable interpretation of which parts of the ECG signal influenced the output most.  
These explanations are saved as images and NumPy arrays for easy viewing and further analysis.  
They provide insight into how the AI focuses on specific waveform features when detecting arrhythmias, helping users and researchers understand the reasoning behind its predictions.

## Installation

1. Clone the repository:
   git clone <repo_url>
   cd <repo_name>

2. Install required dependencies:
   pip install -r requirements.txt

## Running Inference

Run model inference using a configuration file:

   python inference.py --config_file ./configs/inference_config.json --output_dir ./outputs_inference

Predictions are saved to the chosen output directory.

## Running Explainability (XAI)

To generate explanations for model predictions:

   python xai.py --config_file ./configs/xai_config.json --output_dir ./outputs_xai

This script:
- Loads the trained model (either from a checkpoint or a genetic programming structure).
- Runs ensemble explainability on each sample.
- Saves results and visualizations locally.

## Repository Structure

.
├── src/
│   ├── data/                      - Data handling and preprocessing
│   ├── lit_models/                - PyTorch Lightning model definitions
│   ├── utils/                     - Helper and formatting utilities
│   ├── data_model/                - Fold and dataset configurations
│   └── ...
├── configs/                       - JSON configuration files for training, inference, and XAI
├── inference.py                   - Script for running inference
├── xai.py                         - Script for generating explanations
├── requirements.txt               - Required Python packages
└── README.md                      - This file

## Key Components

- **PhysionetDataModule** – Loads and prepares ECG datasets.  
- **GuangzhouLitModel** – Core deep learning model for ECG classification.  
- **InferenceTree (Giraffe)** – Genetic programming-based ensemble that combines models.  
- **Explainer** – Generates combined explanations from multiple attribution methods.

## Outputs

- Predictions: stored as `.pt` tensors containing model outputs and labels.  
- Explanations: saved as `.npz` arrays and `.png` heatmaps for each sample.  

## Citation

If you use this repository, please cite:

**"Interpretable Arrhythmia Detection in ECG Scans Using Deep Learning Ensembles: A Genetic Programming Approach."**
