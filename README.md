# UAP-FedECG

## Uncertainty-Aware and Differentially Private Federated Learning for Non-IID ECG Classification

UAP-FedECG is a research-oriented federated learning system for privacy-preserving electrocardiogram classification across multiple simulated healthcare institutions.

The project investigates whether uncertainty-aware model aggregation can improve federated learning performance when hospitals contain unequal, imbalanced, and non-identically distributed patient data. Differential privacy is also incorporated to reduce information leakage through model updates.

> **Status:** In development  
> **Project type:** University Artificial Intelligence group project  
> **Primary domain:** Federated Learning, Healthcare AI, ECG Classification  
> **Intended use:** Academic research and experimentation only

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Research Question](#research-question)
- [Objectives](#objectives)
- [Key Contributions](#key-contributions)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Hospital Simulation](#hospital-simulation)
- [Model Architecture](#model-architecture)
- [Federated Learning Methods](#federated-learning-methods)
- [Uncertainty-Aware Aggregation](#uncertainty-aware-aggregation)
- [Differential Privacy](#differential-privacy)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Plan](#experimental-plan)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Running the Project](#running-the-project)
- [Configuration](#configuration)
- [Expected Outputs](#expected-outputs)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Ethical and Medical Disclaimer](#ethical-and-medical-disclaimer)
- [Contributors](#contributors)
- [License](#license)

---

## Project Overview

Healthcare institutions often possess valuable patient data that cannot be directly shared because of privacy, legal, and ethical concerns.

Federated learning enables hospitals to collaboratively train a machine learning model without transferring their raw patient records to a centralized server. However, conventional federated learning methods such as Federated Averaging may perform poorly when participating hospitals have different patient populations, class distributions, dataset sizes, and signal quality.

UAP-FedECG addresses this problem by combining:

- Federated ECG classification
- Non-IID hospital data simulation
- Monte Carlo dropout uncertainty estimation
- Uncertainty-aware model aggregation
- Differentially private local training
- Hospital-level performance evaluation

The system uses multiple simulated hospitals created from the PTB-XL ECG dataset. Each hospital trains a local ECG classifier and communicates only model updates and selected statistical information to the federated server.

---

## Problem Statement

Standard Federated Averaging mainly weights each hospital according to its number of training samples.

This can become problematic when a large hospital produces an unreliable update because of:

- Severe class imbalance
- Noisy ECG recordings
- Insufficient local training
- Distribution differences
- Poor calibration
- High predictive uncertainty

A large but unreliable hospital may therefore have more influence than a smaller but reliable hospital.

This project investigates whether predictive uncertainty can be used as an additional reliability signal during federated model aggregation.

---

## Research Question

> Can uncertainty-aware federated aggregation improve ECG classification performance and hospital-level reliability under non-IID data while maintaining measurable privacy through differential privacy?

The project also investigates the following sub-questions:

1. How does non-IID hospital data affect standard FedAvg?
2. Does uncertainty-aware aggregation improve average and worst-hospital performance?
3. How does differential privacy affect classification performance?
4. What is the trade-off between privacy, utility, calibration, and training time?
5. Does uncertainty-based weighting provide a meaningful improvement over sample-count-only aggregation?

---

## Objectives

The main objectives of this project are to:

1. Develop a centralized ECG classification baseline.
2. Train independent local models for simulated hospitals.
3. Implement standard Federated Averaging using Flower.
4. Create non-IID hospital partitions from PTB-XL.
5. Estimate local model uncertainty using Monte Carlo dropout.
6. Develop an uncertainty-aware federated aggregation method.
7. Apply differential privacy using Opacus.
8. Compare centralized, local, federated, uncertainty-aware, and private models.
9. Evaluate hospital-level performance, fairness, calibration, privacy, and efficiency.
10. Produce a reproducible Jupyter or Google Colab implementation.

---

## Key Contributions

The expected contributions of UAP-FedECG are:

- A reproducible non-IID federated ECG classification pipeline
- A compact 1D-CNN suitable for federated simulation
- An uncertainty-aware extension of FedAvg
- Differentially private local ECG training
- Hospital-level performance and fairness analysis
- Calibration and uncertainty evaluation
- A privacy–utility trade-off analysis
- An ablation study of the proposed aggregation mechanism

---

## System Architecture

```text
                 ┌─────────────────────────────┐
                 │     Federated Server        │
                 │                             │
                 │ FedAvg / UA-FedAvg          │
                 │ Global model aggregation    │
                 └──────────────┬──────────────┘
                                │
                  Global model parameters
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  Hospital 1   │       │  Hospital 2   │       │  Hospital 3   │
│               │       │               │       │               │
│ Local ECGs    │       │ Local ECGs    │       │ Local ECGs    │
│ Local 1D-CNN  │       │ Local 1D-CNN  │       │ Local 1D-CNN  │
│ MC Dropout    │       │ MC Dropout    │       │ MC Dropout    │
│ DP Training   │       │ DP Training   │       │ DP Training   │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        └──────── Model updates and uncertainty ──────┘
                                │
                                ▼
                       Secure server-side
                         model aggregation
