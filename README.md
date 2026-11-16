# Evaluation of Machine Learning-Based Network Intrusion Detection Systems under Different Validation Protocols

Artifact accompanying the Master's dissertation by **Gustavo Capelo Gomes**, submitted to the NOVA Information Management School (NOVA IMS), Universidade NOVA de Lisboa, for the Master's Degree in Data Science and Advanced Analytics (specialization in Data Science), July 2026. Supervisor: João Francisco Martins.

This repository contains the complete experimental code, configuration, environment specification, and canonical result artifacts needed to reproduce every table and figure in the dissertation.

---

## Overview

Machine learning-based network intrusion detection systems (NIDS) often report near-ceiling scores on public benchmarks, yet their performance in operational settings is considerably more modest. One likely source of this gap is the protocol used to split benchmark traffic into training and test sets. Random or unspecified splits are common, even though network captures are time-ordered.

This study asks whether random train/test splits inflate reported NIDS performance relative to a temporally realistic evaluation. The comparison holds every element of the pipeline constant except the split protocol, and it is repeated across five random seeds.

## Key Findings

| Dataset | Metric (macro-F1) | Random split (P1) | Temporal holdout (P3) | Inflation gap |
|---|---|---|---|---|
| CIC-IDS2017 | Mean over 5 seeds | 0.991 | 0.340 | 0.651 |
| CSE-CIC-IDS2018 | Mean over 5 seeds | 0.970 | 0.450 | 0.520 |

- **Consistent direction.** All five seeds agree in direction on every metric. The one-sided Wilcoxon signed-rank test yields p = 0.03125, the smallest value attainable with n = 5.
- **Latent-space drift.** Measured in a 16-dimensional autoencoder latent space, 13 to 16 dimensions drift significantly in every seed. Mean Population Stability Index (PSI) is 0.655 on CIC-IDS2017 and 0.227 on CSE-CIC-IDS2018, ordering the two datasets consistently with their performance gaps.
- **Interpretation.** On these benchmarks, the data-split protocol is a first-order determinant of reported performance. Random-split scores should not be read as estimates of forward-in-time generalization.

## Research Design

**Contribution C1: Inflation gap.** The same XGBoost pipeline is evaluated under two protocols applied to identical captures:

- **P1, Stratified random split.** 70% train, 10% validation, 20% test, stratified by label.
- **P3, Temporal holdout.** Splits follow the deployment timeline.
  - CIC-IDS2017: Monday and Tuesday for training, Wednesday for validation, Thursday and Friday for testing. The test window contains attack families unseen during training.
  - CSE-CIC-IDS2018: the first 60% of the ten daily files for training, the next 20% for validation, and the last 20% for testing.

**Contribution C2: Latent-space drift.** An autoencoder is used purely as a dimensionality-reduction instrument, not as an anomaly detector. Each latent dimension is compared between the training and test windows using PSI over training-decile bins and a Bonferroni-corrected Kolmogorov–Smirnov test.

**Fixed pipeline.** Cleaning, train-only preprocessing (imputation, RobustScaler, clipping), selection of the 30 features with highest mutual information, and an XGBoost classifier with fixed hyperparameters are identical across protocols.

| Component | Setting |
|---|---|
| Classifier | XGBoost, 100 trees, max depth 6, learning rate 0.1, subsample and colsample 0.8, early stopping (20 rounds) |
| Feature selection | Top 30 by mutual information, fit on each protocol's training partition |
| Autoencoder | 16-dimensional latent space, 64 hidden units, 50 epochs |
| Seeds | 42, 43, 44, 45, 46 |
| Statistics | One-sided Wilcoxon signed-rank test, 10,000-resample bootstrap confidence intervals, Bonferroni correction for the drift tests |

Binary classification (benign vs. attack) is the only task. Sliding-window protocols, multiclass classification, SHAP-based attribution, and calibration analysis are out of scope.

## Repository Structure

```
.
├── config.py              # All frozen constants: seeds, paths, hyperparameters, split definitions
├── main.py                # Entry point: runs experiments, drift analysis, and reporting
├── requirements.txt       # Pinned environment snapshot (Python 3.12.3)
├── data/                  # Dataset loading, cleaning, and protocol-specific splitting
├── preprocessing/         # Train-only preprocessing pipeline and feature selection
├── models/                # XGBoost classifier wrapper
├── drift/                 # Autoencoder and PSI / KS drift metrics
├── evaluation/            # Classification metrics and statistical tests
├── experiment/            # Experiment runner (dataset × protocol × seed)
├── reporting/             # Table and figure generation
└── results/
    ├── full_run/          # Canonical artifacts underlying every reported number
    │   ├── results_CIC2017_{42..46}.json   # Per-seed audit artifacts, CIC-IDS2017
    │   ├── results_CIC2018_{42..46}.json   # Per-seed audit artifacts, CSE-CIC-IDS2018
    │   ├── results_raw.csv                 # Consolidated per-seed metrics
    │   ├── c1_inflation_gap.csv            # C1 analysis output
    │   ├── c2_drift_summary.csv            # C2 analysis output
    │   ├── c2_psi_per_dim.csv              # Per-dimension latent PSI
    │   ├── tables/                         # Publication tables (hypothesis, split, dataset summary, performance)
    │   └── figures/                        # Figures of Chapter 5
    ├── verify_run_jul10/  # Verification re-run of the CIC-IDS2017 experiments
    └── stale_jun5/        # Earlier run kept for reference; not used for reported numbers
```

The canonical results are in `results/full_run/`. Numbers in the dissertation should be checked against that directory.

## Getting Started

### Requirements

- Python 3.12
- Core packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `scipy`, `torch`, `matplotlib`, `seaborn`

Install from the pinned snapshot:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The snapshot was generated from a development environment and includes packages that the experiments do not use. For a minimal setup, install only the core packages listed above.

### Data

The raw captures are **not redistributed** in this repository. They are available from the Canadian Institute for Cybersecurity. Place them as follows, relative to the repository root:

```
../data/raw/BCCC-CIC-IDS-2017/    # CIC-IDS2017 daily CSV files
../data/raw/CSE-CIC-IDS2018/      # CSE-CIC-IDS2018 daily CSV files
```

The loader reconstructs the working copies from these files. Results are reproducible from the released code.

### Running the Experiments

```bash
# Full run: both datasets, all five seeds
python main.py

# Subset of datasets and seeds
python main.py --datasets CIC2017 --seeds 42 43

# Skip the latent drift analysis
python main.py --skip-drift

# Write outputs to a custom directory
python main.py --output-dir /path/to/results
```

By default, outputs are written to `../results` relative to the repository root. Use `--output-dir` to change this.

## Reproducibility

Reproduction here means that rerunning the released code on the same data yields results within seed-to-seed variation. All constants (split definitions, hyperparameters, seed list, preprocessing thresholds, and statistical settings) are defined in `config.py`. Each experiment runs under the five fixed seeds, and each seed's protocol contrast is paired by seed value.

Bitwise regeneration depends on the pinned environment. The canonical result files in `results/full_run/` are authoritative for every number reported in the dissertation.

## Citation

If you use this work, please cite it as:

```
Gomes, G. C. (2026). Evaluation of Machine Learning-Based Network Intrusion Detection
Systems under Different Validation Protocols [Master's dissertation, NOVA Information
Management School, Universidade NOVA de Lisboa].
```

## Contact

For questions about the study or the code, open an issue in this repository.
