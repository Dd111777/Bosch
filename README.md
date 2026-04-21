# Bosch

A physics-informed learning framework for Bosch DRIE morphology prediction and few-shot transfer.

## Overview

This repository implements a staged learning pipeline for Bosch deep reactive ion etching (DRIE), with a focus on bridging **process recipes**, **IEDF-derived physical descriptors**, and **cycle-resolved morphology prediction**.

Instead of directly fitting etched morphology from recipe parameters alone, the framework introduces an intermediate physical representation (`Phys7`) extracted from ion energy distribution functions (IEDFs), and then uses that representation to condition downstream morphology prediction. The overall workflow supports:

* IEDF parsing and physical descriptor extraction
* Stage A: recipe → Phys7 surrogate learning
* Stage B: Phys7-conditioned morphology sequence prediction
* Stage C: few-shot transfer from simulation-trained models to sparse experimental metrology

The codebase is mainly organized around Bosch TSV / trench etch cases stored in Excel sheets and per-case IEDF CSV files.

## Pipeline

### 1. IEDF-derived physical representation

`extract_phys7_from_iedf.py` parses simulated IEDF files and converts them into a compact 7-dimensional physical descriptor vector (`Phys7`). The current implementation uses dominant ions from:

* `SF6 / sheath2`: `F_1p`, `SF3_1p`, `SF4_1p`, `SF5_1p`
* `C4F8 / sheath1`: `CF3_1p`, `C2F3_1p`

The extracted features are:

* `logGamma_SF6_tot`
* `pF_SF6`
* `spread_SF6`
* `qskew_SF6`
* `logGamma_C4F8_tot`
* `rho_C4F8`
* `spread_C4F8`

The script can also generate IEDF + CDF visualization figures for selected cases.

### 2. Stage A: recipe → Phys7

`stageA_train_phys_pycharm.py` trains a surrogate from the 7-dimensional recipe vector

`[APC, source_RF, LF_RF, SF6, C4F8, DEP time, etch time]`

to the 7-dimensional Phys7 target. The script supports Transformer / GRU / MLP baselines and exports parity plots, residual plots, heatmaps, and prediction tables.

### 3. Stage B: Phys7-conditioned morphology prediction

`stageB_train_morph_on_phys7_pycharm.py` trains the morphology decoder. It uses:

* recipe-derived static features
* Stage A predicted Phys7
* cycle indices / time steps

to predict cycle-resolved morphology targets. The code supports multiple backbones (`transformer`, `gru`, `mlp`), multiple recipe feature augmentation modes (`base`, `time`, `gas`, `rf`, `squares`), and multiple Phys7 ablation modes (`full`, `only_energy`, `only_flux`, `none`).

### 4. Stage C: few-shot transfer on sparse experimental data

`stageC_finetune_joint_on_new_pycharm.py` adapts the pretrained Stage B model to sparse experimental metrology. The transfer stage supports:

* sparse masked supervision
* progressive fine-tuning
* L2-SP regularization
* input adapters for static recipe features and Phys7 sequence calibration
* experiment management over multiple split candidates and fine-tuning configurations

This stage is designed for simulation-to-experiment adaptation under limited measurement labels.

## Repository Structure

```text
Bosch/
├── extract_phys7_from_iedf.py
├── phys_model.py
├── physio_util.py
├── stage0_train_iedf_ae.py
├── stageA_train_phys_pycharm.py
├── stageB_train_morph_on_phys7_pycharm.py
├── stageB_util.py
├── stageB_explain_analysis.py
├── stageC_finetune_joint_on_new_pycharm.py
├── compare_meas_vs_sim.py
├── scientific_data_analysis.py
├── scallop_geom.py
├── case.csv
└── run_all_experiments.py
```


如果你想，我可以下一条直接给你一版**更适合 GitHub 首页展示的精简版 README**。
