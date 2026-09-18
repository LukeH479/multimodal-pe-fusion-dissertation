# Multimodal Fusion for Pulmonary Embolism Diagnosis and Prognosis

Code and experimental results for a multimodal machine-learning study of pulmonary embolism (PE) diagnosis and 12-month mortality prediction using CT imaging and structured electronic health record (EHR) data.

This repository accompanies my third-year Computer Science dissertation at the University of Warwick:

**Multimodal Fusion Strategies for Pulmonary Embolism Diagnosis and Prognosis Using Heterogeneous CT Architectures**

The project investigates whether the stage at which EHR information is combined with CT predictions affects multimodal performance. It compares heterogeneous 2D, 2.5D and 3D CT models, structured EHR models, CT-only ensembles, and two multimodal fusion strategies.

## Overview

The project evaluates two approaches to combining CT and EHR predictions:

* **Flat fusion:** CT and EHR predictions are combined jointly in a single meta-learning stage.
* **Hierarchical fusion:** CT predictions are first combined into a CT ensemble, which is then fused with the EHR prediction.

The models are evaluated on two tasks:

* Pulmonary embolism diagnosis
* 12-month mortality prediction among PE-positive patients

The primary evaluation metric is area under the precision-recall curve (AUPRC), with AUROC, Brier score and expected calibration error (ECE) also reported.

## Models

### CT models

Three CT representations are used:

* **2D:** ResNet-34 with attention pooling over selected slices
* **2.5D:** Modified ResNet-34 using five-slice inputs
* **3D:** DenseNet121 operating on volumetric CT patches

### EHR models

Structured EHR features are modelled using:

* Logistic Regression
* XGBoost

### Ensemble and fusion models

The repository includes implementations for:

* Mean CT ensemble
* Logistic Regression CT stacker
* XGBoost CT stacker
* Flat CT-EHR fusion
* Hierarchical CT-EHR fusion

Stacked models use out-of-fold predictions during training to avoid information leakage.

## Dataset

The project uses the **INSPECT** dataset, which contains linked CT imaging and structured EHR information for pulmonary embolism diagnosis and prognosis.

The INSPECT dataset itself is **not distributed in this repository**.

Processed CT scans, EHR records and OMOP vocabulary data used during development were stored separately from the source repository.

Some derived metadata and experiment outputs may be included where appropriate.

## Repository Structure

```text
.
├── code/
│   ├── main.py
│   ├── sbatch_scripts/
│   └── src/
│       ├── Base_Models/
│       ├── CT_data_pipelines/
│       ├── CT_ensemble_Models/
│       ├── EHR_data_pipelines/
│       ├── Training/
│       └── selecting_scans_pipeline.py
│
├── datasets/
│   ├── ct_scan_metadata/
│   └── filtered_datasets/
│
├── download_data_scripts/
│
├── Experiment_results/
│   ├── Full_experiment_results_seed_*/
│   ├── summarize_main_metrics.py
│   ├── run_subgroup_comparative_analysis.py
│   └── plot_auroc_auprc_summary.py
│
├── Model_evaluation_metrics/
│
├── job_outputs/
│
└── README.md
```

### `code/`

Contains the main modelling and training code.

`main.py` is the primary entry point for running model experiments.

The `src/` directory contains the CT and EHR pipelines, base models, ensemble models, training utilities and cohort-selection logic.

### `code/sbatch_scripts/`

Slurm job submission scripts used to run preprocessing and model-training jobs on a compute cluster.

### `datasets/`

Contains scan metadata and derived files used for cohort construction and experiment splitting.

It does **not** contain the underlying EHR dataset or processed CT volumes.

### `download_data_scripts/`

Utilities for identifying, batching, aligning and downloading CT scans.

### `Experiment_results/`

Contains saved experiment outputs and scripts used to aggregate and analyse completed runs.

Experiments were repeated across multiple random seeds. Per-seed outputs are stored in `Full_experiment_results_seed_*` directories.

This directory also contains scripts for:

* Metric aggregation
* Subgroup analysis
* AUROC/AUPRC result visualisation

### `Model_evaluation_metrics/`

Intermediate output directory used for individual and batch model runs before results are organised into the main experiment-results directories.

### `job_outputs/`

Slurm standard-output and error logs generated during cluster execution.

## Experimental Workflow

The overall workflow is:

1. Construct the eligible CT cohort using `code/src/selecting_scans_pipeline.py`.
2. Download and preprocess CT scans using `download_data_scripts/`.
3. Construct structured EHR features using `code/src/EHR_data_pipelines/`.
4. Generate multimodal eligibility files and train/test splits.
5. Train individual CT and EHR models.
6. Generate out-of-fold predictions for ensemble and fusion models.
7. Train CT-only and multimodal fusion models.
8. Evaluate models across repeated seeded runs.
9. Aggregate metrics and perform subgroup and interpretability analyses.

## Results

The main findings were:

* Structured EHR models provided strong standalone predictive performance for both tasks.
* CT-only stacking produced modest improvements for PE diagnosis but limited gains for prognosis.
* For diagnosis, flat and hierarchical fusion produced very similar performance.
* For prognosis, hierarchical fusion performed better than flat fusion within the multimodal models.
* The EHR-only XGBoost model remained the strongest overall model for 12-month mortality prediction.

Full experimental results and analyses are presented in the accompanying dissertation.

## Dissertation

The full dissertation is available in this repository under:

```text
dissertation/
└── dissertation.pdf
```

**Luke Hidveghy**
Department of Computer Science
University of Warwick
2025–2026

## Reproducibility

The repository contains the source code and experiment configuration used for the dissertation.

Full reproduction additionally requires access to the INSPECT dataset and the associated clinical-data resources. These data are not included in this repository.

The original experiments were run using Slurm-based compute infrastructure with GPU acceleration for CT model training.

## License

Source code in this repository is licensed under the MIT License unless otherwise stated.

The dissertation PDF is © Luke Hidveghy and is not covered by the MIT License.
