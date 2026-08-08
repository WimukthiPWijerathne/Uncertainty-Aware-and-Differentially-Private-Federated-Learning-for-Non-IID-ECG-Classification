# UAP-FedECG

## Uncertainty-Aware and Differentially Private Federated Learning for Non-IID ECG Classification

> **Course:** Intelligent System Design — University group project
> **Dataset:** PTB-XL (12-lead ECG, 100 Hz)
> **Status:** Experiments complete; final report notebook finished (submission 2026-08-03)

This README is a from-the-ground-up study guide for the whole repository: every concept, every file, every script, how to run each one, what it produces, why the system is built the way it is, and the actual results the team obtained. It is written so you can walk into the evaluation and answer "what does X do / why does X exist" for any part of the codebase.

---

## Table of Contents

1. [30-Second Pitch](#1-30-second-pitch)
2. [Core Concepts (Glossary)](#2-core-concepts-glossary)
3. [Problem Statement & Research Question](#3-problem-statement--research-question)
4. [Dataset](#4-dataset)
5. [Hospital Simulation (Non-IID Partitioning)](#5-hospital-simulation-non-iid-partitioning)
6. [System Architecture](#6-system-architecture)
7. [Model Architecture](#7-model-architecture)
8. [Repository Structure](#8-repository-structure)
9. [Source Code Walkthrough (`src/`)](#9-source-code-walkthrough-src)
10. [Pipeline Walkthrough (`scripts/`)](#10-pipeline-walkthrough-scripts)
11. [Federated Learning Methods, Explained](#11-federated-learning-methods-explained)
12. [Differential Privacy, Explained](#12-differential-privacy-explained)
13. [Final Results (Actual Numbers)](#13-final-results-actual-numbers)
14. [Discussion — Why the Results Look Like This](#14-discussion--why-the-results-look-like-this)
15. [Limitations](#15-limitations)
16. [How to Run Everything, End to End](#16-how-to-run-everything-end-to-end)
17. [Known Issues / Bugs to Check Before Demoing](#17-known-issues--bugs-to-check-before-demoing)
18. [Architectural Decisions — Q&A](#18-architectural-decisions--qa)
19. [Evaluation-Day Cheat Sheet](#19-evaluation-day-cheat-sheet)
20. [Notebook Guide](#20-notebook-guide)
21. [Tech Stack](#21-tech-stack)
22. [References](#22-references)

---

## 1. 30-Second Pitch

Hospitals have ECG data they legally/ethically can't pool together. **Federated learning (FL)** lets them jointly train one shared model by exchanging only model weights, never raw patient records. But real hospitals don't have similar data (**non-IID**: different disease mixes, different dataset sizes) — and standard FL (`FedAvg`) can let one confidently-wrong hospital drag the whole model down.

This project:
1. Simulates **4 non-IID hospitals** out of the PTB-XL ECG dataset (Dirichlet-skewed, patient-safe).
2. Trains a compact **1D-CNN** to classify each ECG into 5 diagnostic superclasses (multi-label): `NORM, MI, STTC, CD, HYP`.
3. Compares **6 training regimes**: Centralized, Standard FedAvg, "Compatible" FedAvg (control), Uncertainty-Aware FedAvg (this project's contribution), DP-FedAvg, DP-UA-FedAvg.
4. Estimates each hospital's prediction uncertainty using **Monte Carlo Dropout**, and tests whether weighting the aggregation by uncertainty (instead of just sample count) helps.
5. Adds **Differential Privacy** (Opacus DP-SGD) to bound how much any single patient's ECG can influence the shared model, and measures the privacy/accuracy trade-off (ε).
6. Shows that **post-hoc, validation-only threshold calibration** can recover a huge chunk of the accuracy DP destroys — without retraining and without spending any extra privacy budget.

---

## 2. Core Concepts (Glossary)

If you only remember one section before the viva, remember this one.

| Term | Plain-English meaning |
|---|---|
| **Federated Learning (FL)** | Training one shared ("global") model across multiple parties (here, hospitals) without any party ever sending its raw data anywhere. Only model weights move. |
| **Client** | One participant in FL. Here: one simulated hospital, with its own private ECG partition and its own local copy of the model. |
| **Global model** | The one model that lives conceptually "on the server." Sent to clients → clients train it locally → clients send back updated weights → server averages them → repeat. |
| **Communication round** | One full cycle of the above (send → train → return → aggregate). This project runs up to 50 rounds per experiment, with early stopping. |
| **FedAvg (Federated Averaging)** | The standard aggregation rule: `w_global = Σ(n_k · w_k) / Σ n_k` — each client's weights are averaged, weighted by how many training samples that client has. |
| **IID vs. Non-IID** | IID = every client's data looks statistically the same. Non-IID = it doesn't (different label proportions, different amounts of data, different signal characteristics). Real hospitals are always non-IID. This is the central difficulty FL research targets. |
| **Label skew** | One flavor of non-IID: different class proportions per client (e.g., one hospital sees mostly STTC cases, another mostly CD). This project's hospitals were generated to have exactly this. |
| **Quantity skew** | Different amounts of data per client. In this project: Hospital 0 has 8,460 records, Hospital 3 has only 1,344. |
| **Client drift** | The failure mode where averaging models trained on very different local data distributions makes the *global* model worse than some *individual* local models, because clients effectively learned to solve slightly different problems. |
| **Predictive (epistemic) uncertainty** | How stable a model's prediction is for a given input if you perturb it slightly — not "is it right" but "would it keep saying the same thing." |
| **Dropout** | Training-time regularization: randomly zero out a fraction of neurons every forward pass so the network can't over-rely on any single one. Normally disabled at inference. |
| **Monte Carlo (MC) Dropout** | Trick for estimating uncertainty: keep dropout **on** at inference and run the same input through the network N times. Each pass drops different neurons, giving slightly different probabilities. The spread (variance/entropy) across those N passes *is* the uncertainty estimate. |
| **Uncertainty-Aware FedAvg (UA-FedAvg)** | This project's proposed aggregation rule: weight each client not only by sample count but also by `1 / uncertainty`, so confident/stable hospitals get more say and shaky ones get down-weighted — regardless of how much data they have. |
| **Differential Privacy (DP)** | A mathematical guarantee bounding how much any single training record could have changed the model's output. Implemented here via **DP-SGD**: clip each per-example gradient to a fixed norm, then add calibrated Gaussian noise before the optimizer step. |
| **Gradient clipping** | Capping the L2 norm of each *individual* example's gradient so one outlier record can't dominate an update. Prerequisite for DP-SGD's privacy proof. |
| **Epsilon (ε) / Delta (δ)** | The "privacy budget." Smaller ε = more noise = stronger privacy = worse accuracy. δ is the (tiny) probability the guarantee fails. This project reports ε (max ≈5.46) at δ=1e-5. |
| **Privacy accountant** | The bookkeeping algorithm that tracks how much privacy budget (ε) has been spent as training progresses. This project uses Opacus's `"prv"` accountant. |
| **Opacus** | The PyTorch library used to implement DP-SGD (gradient clipping + noise + accounting) for local hospital training. |
| **Flower (flwr)** | A popular FL orchestration framework. It's listed as a dependency, but **this project does not actually use it** — see [§18](#18-architectural-decisions--qa) for why. |
| **PTB-XL** | The ECG dataset used: ~21k 12-lead recordings, a standard cardiology ML benchmark, with an official patient-safe train/validation/test fold. There's no real hospital-of-origin field, so "hospitals" here are synthetically partitioned. |
| **1D-CNN** | A convolutional network adapted for 1-D time-series (ECG leads over time) instead of 2-D images. |
| **Multi-label classification** | Each ECG can belong to more than one of the 5 classes at once (e.g., both MI and STTC). Modeled with 5 independent sigmoid outputs + `BCEWithLogitsLoss`, not softmax. |
| **Macro-F1 / Weighted-F1 / AUROC** | Macro-F1 = F1 averaged equally across the 5 classes (penalizes ignoring rare classes). Weighted-F1 = F1 averaged by class frequency. AUROC = ranking quality, independent of the decision threshold. |
| **Threshold calibration** | Choosing a separate decision threshold per class (instead of a fixed 0.5) using only validation data, to convert probabilities into 0/1 predictions more sensibly — especially important when DP noise collapses probabilities toward one side. |

---

## 3. Problem Statement & Research Question

Standard FedAvg weights every hospital purely by how many training samples it has. A large hospital with **noisy, imbalanced, or under-trained** data can therefore dominate the global model even if its update is unreliable, while a smaller-but-cleaner hospital gets drowned out.

**Research question:**
> Can uncertainty-aware federated aggregation improve ECG classification performance and hospital-level reliability under non-IID data, while maintaining measurable privacy through differential privacy?

Sub-questions actually answered by the experiments in this repo:
1. How much does non-IID hospital data hurt standard FedAvg relative to centralized training? → **Small hit** (0.6333 vs 0.6686 Macro-F1, −5.3%).
2. Does uncertainty-aware aggregation improve on sample-count-only aggregation? → **No**, not on this partition (see §14).
3. How much does differential privacy cost in accuracy? → **Large** at fixed threshold (0.6333 → 0.2750 Macro-F1), but AUROC stays high (0.77), which motivated calibration.
4. Can you recover utility without spending more privacy budget? → **Yes** — calibration alone: 0.2750 → 0.5341 Macro-F1.

---

## 4. Dataset

**PTB-XL** — 12-lead ECGs at 100 Hz, 1000 samples (10 seconds) per lead, from PhysioNet.

- Raw files live in `data/raw/ptb-xl/` (never committed — everyone downloads it locally; see `.gitignore`).
- Each record's SCP-ECG diagnostic codes are mapped to **5 diagnostic superclasses**: `NORM` (normal), `MI` (myocardial infarction), `STTC` (ST/T change), `CD` (conduction disturbance), `HYP` (hypertrophy).
- This is **multi-label**: one ECG can carry more than one superclass positive at once, so the target is a 5-element multi-hot vector, not a single class index.
- Records with none of the 5 superclasses are dropped.

After filtering, the processed dataset has **21,388 ECGs**, split using PTB-XL's **official patient-safe stratified folds** (so no patient's recordings straddle a split):

| Split | ECGs | Patients | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 17,084 | 14,823 | 7,596 | 4,379 | 4,186 | 3,907 | 2,119 |
| Validation | 2,146 | 1,917 | 955 | 540 | 528 | 495 | 268 |
| Test | 2,158 | 1,877 | 963 | 550 | 521 | 496 | 262 |

Fold assignment rule (see `scripts/02_prepare_metadata.py`): `strat_fold` 1–8 → train, 9 → validation, 10 → test.

Each waveform is loaded on demand via `wfdb.rdsamp`, non-finite values are zeroed, and the array is transposed from WFDB's `(1000 samples, 12 leads)` to the CNN's expected `(12 channels, 1000 samples)`.

---

## 5. Hospital Simulation (Non-IID Partitioning)

Only the **training** split (17,084 records / 14,823 patients) is partitioned into 4 simulated hospitals — validation and test stay shared/global so every method is judged on the same, untouched data.

Method (`scripts/07_create_hospital_partitions.py`):
1. Each **patient** (not ECG) is assigned one *primary class* — whichever of the 5 superclasses occurs most often across that patient's own ECGs (ties broken toward the globally rarer class).
2. Within each primary-class group, patients are shuffled and split across the 4 hospitals using a **Dirichlet(α=0.5)** distribution — small α means *more* skew between hospitals.
3. All ECGs belonging to an accepted patient go to that patient's assigned hospital (never split across hospitals) — this is what makes it "patient-level" partitioning, avoiding leakage.
4. An assignment is only accepted if every hospital ends up with ≥100 patients (retried up to 100 times with different random draws otherwise).
5. Random seed fixed at 42 for reproducibility.

Resulting partitions (verified by `scripts/08_verify_hospital_partitions.py`, inspected by `scripts/08b_inspect_non_iid.py`):

| Hospital | Records | Patients |
|---|---:|---:|
| Hospital 0 | 8,460 | 7,328 |
| Hospital 1 | 3,931 | 3,488 |
| Hospital 2 | 3,349 | 2,919 |
| Hospital 3 | 1,344 | 1,088 |

Class prevalence per hospital (this is the actual non-IID skew being studied):

| Hospital | NORM | MI | STTC | CD | HYP |
|---|---:|---:|---:|---:|---:|
| Hospital 0 | 46.1% | 29.6% | 27.2% | 5.4% | 14.5% |
| Hospital 1 | 52.2% | 18.8% | 11.7% | 43.3% | 10.6% |
| Hospital 2 | 46.7% | 24.8% | 13.2% | 42.4% | 9.3% |
| Hospital 3 | 5.7% | 22.9% | **73.1%** | 24.1% | 12.0% |

Hospital 3 is simultaneously the **smallest** hospital and the most **skewed** (dominated by STTC, almost no NORM) — the hardest client in the study, and (as shown in §13) the worst-performing standalone local model.

---

## 6. System Architecture

```text
                 ┌─────────────────────────────┐
                 │     Federated Server        │
                 │ FedAvg / UA-FedAvg          │
                 │ Global model aggregation    │
                 └──────────────┬──────────────┘
                                │  global model parameters
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  Hospital 0-3 │       │  Hospital ...  │      │  Hospital ...  │
│ Local ECGs    │       │ Local ECGs    │       │ Local ECGs    │
│ Local 1D-CNN  │       │ Local 1D-CNN  │       │ Local 1D-CNN  │
│ MC Dropout    │       │ MC Dropout    │       │ MC Dropout    │
│ (optional) DP │       │ (optional) DP │       │ (optional) DP │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        └──────── updated weights + n_samples + uncertainty ──────┘
                                │
                                ▼
                   weighted aggregation → new global model
```

Per-round flow, concretely (this is what every `scripts/1X_run_*.py` does):
1. Server holds `global_parameters` (a list of NumPy arrays — the CNN's `state_dict` values).
2. For each hospital client: load `global_parameters` into a fresh local model copy → train 1 local epoch on that hospital's private data only → (if uncertainty-aware) run MC-dropout on the shared validation set to estimate that hospital's uncertainty → (if DP) this local training used DP-SGD → return updated weights + sample count (+ uncertainty).
3. Server aggregates all clients' returned weights into a new `global_parameters` (plain FedAvg = sample-weighted average; UA-FedAvg = sample-and-uncertainty-weighted average).
4. Global model is evaluated on the **shared** validation split; if it improved, checkpoint it; track early-stopping patience.
5. Repeat up to `MAX_ROUNDS` (50) or until `EARLY_STOPPING_PATIENCE` (5) rounds pass with no improvement.
6. Load the best checkpoint and do one final evaluation on the **shared, held-out** test split.

This whole loop is implemented **manually in each script** — it is *not* built on Flower's client/server simulation runtime (see [§18](#18-architectural-decisions--qa)).

---

## 7. Model Architecture

`src/models/cnn1d.py` — `ECG1DCNN`. Same architecture used everywhere (centralized, all FedAvg variants, DP variants); DP training additionally validates it through Opacus's `ModuleValidator`.

Input: `(batch, 12 channels, 1000 samples)` → Output: `(batch, 5)` raw logits (sigmoid applied outside the model, in the loss/eval code).

```text
Conv1d(12 → 32, kernel=7, pad=3, bias=False) → GroupNorm(8, 32) → ReLU → MaxPool1d(2)
Conv1d(32 → 64, kernel=5, pad=2, bias=False) → GroupNorm(8, 64) → ReLU → MaxPool1d(2)
Conv1d(64 → 128, kernel=3, pad=1, bias=False) → GroupNorm(8, 128) → ReLU
AdaptiveAvgPool1d(1)
Dropout(p=0.3)
Linear(128 → 5)
```

- **38,597 total parameters** (tiny, by design — see §18 for why).
- **GroupNorm, not BatchNorm** — BatchNorm's running statistics are incompatible with DP-SGD's per-example gradients (Opacus's `ModuleValidator` would reject it); GroupNorm normalizes within one sample, so it works identically with or without DP.
- **Dropout stays in the module** so it can be reused for MC-Dropout at inference (`enable_mc_dropout()` in the same file flips only dropout layers back to train-mode while everything else stays in eval-mode).
- Loss: `nn.BCEWithLogitsLoss()` (independent binary decision per class — correct choice for multi-label, unlike softmax/cross-entropy which assumes mutually exclusive classes).

---

## 8. Repository Structure

```text
uap-fedecg/
├── README.md                     ← you are here
├── setup.md                      ← quick pip/venv setup guide
├── pyproject.toml / uv.lock      ← uv-managed dependencies (Python 3.12)
├── requirements.txt              ← plain pip alternative
├── main.py                       ← placeholder entry point ("Hello from ...") — not used by the pipeline
│
├── data/
│   ├── raw/ptb-xl/               ← downloaded PTB-XL waveforms + metadata (gitignored)
│   ├── processed/                ← ptbxl_superclasses.csv, class_names.txt (generated, gitignored)
│   └── partitions/                ← hospital_{0..3}.csv, partition summary (generated, gitignored)
│
├── docs/
│   ├── UAP-FedECG_notes.md       ← glossary (source for §2 above)
│   └── uap_fedecg_architecture.png
│
├── notebooks/
│   ├── Final_ECG_FL_Project_Notebook.ipynb   ← THE final report notebook (see §20)
│   ├── Final_ECG_FL_Project_Notebook.html    ← rendered/exported version
│   ├── 01_data_exploration.ipynb             ← empty/starter
│   └── final_project_notebook.ipynb          ← empty/starter
│
├── src/
│   ├── data/          ← PTB-XL PyTorch Dataset + path resolution (§9)
│   ├── models/         ← ECG1DCNN (§7)
│   ├── training/        ← generic train_one_epoch / evaluate_model (§9)
│   ├── uncertainty/       ← public MC-dropout API (§9)
│   └── federated/        ← every FL client type + aggregation strategy (§9, §11)
│
├── scripts/             ← 19 numbered, run-in-order pipeline scripts (§10)
│
├── results/
│   ├── checkpoints/      ← saved model weights per experiment (gitignored except examples)
│   ├── logs/             ← per-round / per-epoch CSV histories (gitignored)
│   ├── figures/          ← training curves, confusion matrices, ROC curves (gitignored)
│   ├── tables/           ← test-metric CSVs per experiment (gitignored)
│   ├── predictions/       ← saved example predictions + model summaries (a few committed)
│   └── threshold_calibration/dp_fedavg/  ← calibrated thresholds + before/after comparison (committed)
│
└── tests/               ← currently empty (only a .gitkeep)
```

> **Note on what's actually in your local checkout vs. git:** `data/raw`, `data/processed`, `data/partitions`, and most of `results/` are in `.gitignore` — every teammate generates them locally by running the scripts. The exceptions that *are* committed are the DP-FedAvg calibration outputs and a few example predictions (used directly by the final notebook), plus whatever you've personally already run on this machine (e.g., `centralized_best.pt` and its figures exist locally right now but are **not** tracked by git).

---

## 9. Source Code Walkthrough (`src/`)

### `src/data/paths.py`
Centralizes every path the project uses (`PROJECT_ROOT`, `DATA_DIR`, `PTBXL_ROOT`, `RESULTS_DIR`, etc.) so scripts never hardcode paths. `get_dataset_root()` resolves the PTB-XL location (overridable via the `PTBXL_DATASET_ROOT` environment variable) and raises a clear `FileNotFoundError` listing exactly which required files/folders (`ptbxl_database.csv`, `scp_statements.csv`, `records100/`) are missing.

### `src/data/ecg_dataset.py`
`PTBXLDataset(Dataset)` — lazily loads one ECG waveform per `__getitem__` call (nothing is pre-loaded into memory). Validates the metadata has the right label columns, checks the `.hea` header file exists before reading, asserts the loaded shape is exactly `(1000, 12)`, replaces NaN/inf with 0, transposes to `(12, 1000)`, and optionally z-score normalizes per-record per-lead (`normalize_per_record`, disabled by default in every experiment). Returns `(signal_tensor, label_tensor)` — a `(12, 1000)` float tensor and a `(5,)` multi-hot float tensor.

### `src/models/cnn1d.py`
Covered in §7. Also hosts the **MC-dropout machinery**: `enable_mc_dropout()` (flip only dropout sub-modules to train-mode), `mc_dropout_predict()` (run N stochastic passes, return mean probabilities + a dict of uncertainty metrics: `variance`, `predictive_entropy`, `expected_entropy`, `mutual_information` — all normalized to roughly `[0, 1]` by dividing by `ln(2)`), `binary_entropy()` (Bernoulli entropy per class, since this is multi-label not single-label), and `hospital_uncertainty_score()` (sample-weighted mean uncertainty across a whole DataLoader — used to score a hospital as a single number).

### `src/training/train.py` and `evaluate.py`
Framework-agnostic (no FL awareness). `train_one_epoch()` runs one epoch of standard supervised training, checks the loss is finite (raises immediately if not — fail fast rather than silently training on NaN). `evaluate_model()` runs a full pass with no gradients, computes macro/weighted F1 at a given decision threshold, per-class F1, per-class AUROC (skipping classes with only one label value present, since AUROC is undefined there), and returns the raw labels/probabilities too (so calibration scripts can recompute F1 at other thresholds without re-running the model).

### `src/uncertainty/mc_dropout.py`
A thin public-facing re-export of the MC-dropout functions from `cnn1d.py` — exists so other code can `from src.uncertainty import mc_dropout_predict` instead of reaching into the model module.

### `src/federated/parameter_utils.py`
The glue between PyTorch and "FL parameters as plain arrays": `get_model_parameters()` (state_dict → `list[np.ndarray]`) and `set_model_parameters()` (list of arrays → state_dict, with strict shape/name checking). Every aggregation function operates purely on these NumPy lists, decoupled from any specific `nn.Module`.

### `src/federated/client.py` — `HospitalClient`
The base, **non-private, non-uncertainty-aware** hospital client. Owns its own `ECG1DCNN`, its own train/validation `DataLoader`s (built only from that hospital's partition + the shared validation split), and an `Adam` optimizer. `.fit(global_parameters)` loads the global weights, trains `local_epochs` epochs (1, in every experiment here), evaluates on the shared validation set, and returns `(updated_parameters, num_train_samples, metrics_dict)` — the exact 3-tuple every aggregation function expects.

### `src/federated/fedavg.py` — `aggregate_fedavg()`
Plain sample-weighted FedAvg: `Σ (n_k/N) · w_k`, done in float64 internally for numerical stability then cast back to the original dtype. Validates every client returned the same number of parameter tensors with matching shapes before combining.

### `src/federated/ua_client.py` — `UncertaintyAwareHospitalClient(HospitalClient)`
Subclasses the base client, keeps its `.fit()` behavior identical, and **adds** an MC-dropout uncertainty pass (`estimate_predictive_uncertainty`, `mc_passes=10` by default) on the shared validation loader after local training, then stuffs `metrics["uncertainty"]` into the returned dict.

### `src/federated/uncertainty.py` — `estimate_predictive_uncertainty()`
The uncertainty estimator actually used by the federated clients (a separate, slightly simpler implementation from `cnn1d.mc_dropout_predict`, focused purely on returning one scalar: mean predictive *variance* across all samples/classes/MC passes). Carefully puts the whole model in `eval()` first (so GroupNorm/BatchNorm stats don't change) then re-enables *only* dropout layers, runs `mc_passes` stochastic forward passes, and averages the variance.

### `src/federated/ua_fedavg.py` — `aggregate_uncertainty_fedavg()`
This project's core proposed aggregation rule:

```
raw_weight_k    = n_k / (u_k + epsilon)
final_weight_k  = raw_weight_k / Σ_j raw_weight_j
```

i.e., same numerator as FedAvg (sample count) but divided by that client's uncertainty score, so a confident hospital's per-sample influence is amplified and an uncertain hospital's is dampened. Non-floating-point tensors (e.g., a BatchNorm `num_batches_tracked` counter) can't be weight-averaged sensibly, so those are instead just copied whole from whichever client got the single largest weight. Returns the aggregated parameters *and* both the normalized and raw weights (so experiment scripts can log exactly how much influence each hospital got, round by round).

### `src/federated/dp_client.py` — `DPHospitalClient`, `create_dp_compatible_ecg_model()`
The private client. `create_dp_compatible_ecg_model()`:
1. Builds a fresh `ECG1DCNN`.
2. Calls `disable_inplace_activations()` — turns off any `inplace=True` on ReLU-family modules, because Opacus's per-sample-gradient backward hooks would otherwise conflict with in-place tensor mutation (this codebase's `ECG1DCNN` uses `inplace=True` ReLUs by default; DP training needs a non-in-place copy).
3. Runs `ModuleValidator.fix()` (would swap out anything DP-incompatible, e.g. BatchNorm→GroupNorm — a no-op here since the model already uses GroupNorm) then `ModuleValidator.validate(strict=True)` to hard-fail if anything is still incompatible.

`DPHospitalClient.__init__` wraps the model/optimizer/dataloader through `PrivacyEngine.make_private()` — this is where `noise_multiplier`, `max_grad_norm` (clipping), `poisson_sampling=True` (privacy accounting requires proper Poisson-sampled minibatches, not simple shuffling), and `grad_sample_mode="hooks"` (per-example gradient computation via backward hooks) are all configured. `.fit()` mirrors the base client but additionally clears Adam's optimizer state each round (to match the non-DP client's "fresh optimizer per round" behavior) and reports the hospital's current cumulative `epsilon` via `privacy_engine.get_epsilon(delta)`.

### `src/federated/dp_ua_client.py` — `DPUAHospitalClient(DPHospitalClient)`
Combines DP-SGD *and* uncertainty estimation: identical pattern to `ua_client.py`, but subclassing the **private** client instead of the plain one. Runs MC-dropout after each private local update and validates the resulting uncertainty is finite and non-negative before returning it (fails loudly rather than silently aggregating garbage).

### `src/federated/compatible_client.py` — `CompatibleHospitalClient`
A **control/ablation** client: uses the exact same DP-compatible architecture (`create_dp_compatible_ecg_model`) as the DP clients, but trains it **without** any privacy engine — plain SGD via Adam, no clipping, no noise. This isolates one specific question: *is any performance drop in the private experiments caused by the architecture change (GroupNorm, non-inplace ReLU) or by the actual DP noise/clipping?* (Answer, from §13/§14: architecture change alone costs nothing — Compatible FedAvg matches Standard FedAvg to 4 decimal places.)

### `src/federated/server_app.py`
```python
"""Flower server application will be implemented here."""
```
Literally just that docstring — an intentionally unused stub. See §18.

---

## 10. Pipeline Walkthrough (`scripts/`)

Run these **in numeric order** the first time; after that, only the ones you need. All scripts assume the project root is on `sys.path` (they add it themselves) and read from `src/data/paths.py`'s resolved locations. Run with `uv run python scripts/NN_name.py` (or plain `python scripts/NN_name.py` inside an activated venv — see §16).

| # | Script | What it does | Key output |
|---|---|---|---|
| 01 | `01_verify_ptbxl.py` | Confirms every `.hea`/`.dat` waveform file referenced in `ptbxl_database.csv` actually exists on disk (both 100 Hz and 500 Hz pairs). | Console pass/fail only. |
| 02 | `02_prepare_metadata.py` | Parses each record's `scp_codes`, maps them through `scp_statements.csv`'s `diagnostic_class` to the 5 superclasses, drops unlabeled records, assigns `split` from `strat_fold`. | `data/processed/ptbxl_superclasses.csv`, `data/processed/class_names.txt` |
| 03 | `03_smoke_test_dataset.py` | Loads one sample and one batch through `PTBXLDataset` + `DataLoader`, asserts shapes are exactly `(12,1000)` / `(8,12,1000)` etc. | Console pass/fail — this is the "Dataset smoke test passed" line from `setup.md`. |
| 04 | `04_smoke_test_model.py` | Forward + backward pass of `ECG1DCNN` on one real batch, prints logits/probability shapes and initial loss. | Console output — the "Forward pass output shape..." lines from `setup.md`. |
| 05 | `05_train_centralized.py` | Trains the **centralized baseline**: all training records pooled, up to 50 epochs, early stopping (patience 5). ⚠️ **Currently broken as committed — see §17.** | `results/checkpoints/centralized_best.pt` / `_last.pt`, `results/logs/centralized_history.csv`, loss/F1 figures |
| 06 | `06_evaluate_centralized.py` | Loads `centralized_best.pt`, evaluates once on the held-out **test** split, prints + saves per-class F1/AUROC. | `results/tables/centralized_test_metrics.csv` |
| 07 | `07_create_hospital_partitions.py` | Builds the 4 non-IID hospital partitions described in §5 (Dirichlet α=0.5, seed 42, patient-level, retried until valid). | `data/partitions/hospital_{0..3}.csv`, `hospital_partition_summary.csv`, class-count/prevalence figures |
| 08 | `08_verify_hospital_partitions.py` | Hard integrity checks: no ECG or patient duplicated across hospitals, every original training ECG assigned exactly once, no validation/test leakage. | Console pass/fail. |
| 08b | `08b_inspect_non_iid.py` | Sanity-checks that the partitions are non-IID *enough* (minimum records/patients/positives per hospital, minimum prevalence range across hospitals for at least 2 classes) — otherwise warns you to regenerate with a different seed/α. | Console report + warnings. |
| 09 | `09_train_local_baselines.py` | Trains **one independent model per hospital** (fresh init each time, no federation at all) — the "what if each hospital just trained alone" baseline. Same shared validation/test split as everything else, for fair comparison. | `results/checkpoints/local_earlystop_max_50/hospital_X_best.pt`, per-hospital + combined epoch histories, `local_earlystop_max_50_results.csv` |
| 10 | `10_run_fedavg.py` | The **manual FedAvg** experiment (§6's loop, `aggregate_fedavg`). Up to 50 rounds, early stopping. | `results/checkpoints/fedavg_.../fedavg_best.pt`, round/client history CSVs, final test-results CSV |
| 11 | `11_run_uncertainty_fedavg.py` | Same loop but with `UncertaintyAwareHospitalClient` + `aggregate_uncertainty_fedavg` (MC passes=10). Logs each hospital's `sample_only_weight` vs. final `UA weight` every round — directly shows how much the uncertainty term is reweighting each client. | `results/checkpoints/ua_fedavg_.../ua_fedavg_best.pt` + histories/results |
| 12 | `12_run_dp_fedavg.py` | DP-FedAvg: `DPHospitalClient` (Opacus DP-SGD, noise=1.0, clip=1.0, δ=1e-5, accountant=`prv`) + plain `aggregate_fedavg`. | `results/checkpoints/dp_fedavg_.../dp_fedavg_best.pt` + histories/results (includes per-round ε) |
| 12b | `12_run_dp_fedavg_smoke_test.py` | Cut-down/quick version of script 12 for verifying the DP pipeline runs before committing to a full 50-round run. | Same shape of outputs, smaller run. |
| 12c | `12_run_dp_fedavg_completed_noise_1_backup.py` | Archived copy of the completed noise=1.0 DP-FedAvg run (kept as a backup so the noise sweep script doesn't need to redo it). | — |
| 13 | `13_run_compatible_fedavg.py` | **Control experiment**: `CompatibleHospitalClient` (same DP-ready architecture, no privacy engine) + plain `aggregate_fedavg`. Isolates architecture effects from DP effects. | `results/checkpoints/compatible_fedavg_.../compatible_fedavg_best.pt` + histories/results |
| 14 | `14_run_dp_noise_sweep.py` | Re-runs script 12 as a subprocess for several `noise_multiplier` values (default `0.75, 1.25, 1.50`; skips ones whose result CSV already exists unless `--force`) to trace the privacy/utility curve. | One `..._test_results.csv` per noise level |
| 16 | `16_run_dp_ua_fedavg.py` | **DP-UA-FedAvg**: `DPUAHospitalClient` (DP-SGD **and** MC-dropout uncertainty) + `aggregate_uncertainty_fedavg`. Full CLI (`--max-rounds`, `--mc-passes`, `--noise-multiplier`, etc.) — most configurable of all the runners; also writes a machine-readable summary JSON and a live run-status JSON updated after every round. | Checkpoints, CSV/JSON logs, final summary JSON |
| 17 | `17_finalize_project_results.py` | Combines the DP-UA summary JSON with the already-validated baseline numbers into one final comparison table + the report's key figures. Run only after every other experiment has completed. | `results/tables/final_model_comparison.csv`, `results/figures/final_project/*`, source manifest JSON |
| 18 | `18_show_model_predictions.py` | Loads a **calibrated** checkpoint (default: the calibrated DP-FedAvg one), prints the architecture summary, runs it on sample test ECGs, saves example plots + a predictions CSV. Supports either the checkpoint's stored per-class thresholds or one manual `--threshold`. | `results/predictions/<name>/{model_summary.txt, sample_predictions.csv, example_ecg.png, example_probabilities.png}` |
| 19 | `19_calibrate_class_thresholds.py` | **Threshold calibration** (no retraining): sweeps candidate thresholds (0.01→0.99, step 0.01) per class *on the validation split only*, picks the F1-maximizing threshold per class, freezes it, then reports the *test*-set F1/AUROC before vs. after. | `results/threshold_calibration/<method>/{calibration_summary.json, class_thresholds.csv, calibrated_thresholds.png, checkpoint_with_calibrated_thresholds.pt, test_per_class_comparison.csv, ...}` |

Notes:
- **Every experiment evaluates on the exact same shared validation/test split** — this is what makes the final comparison table in §13 fair.
- All the FedAvg-family scripts (10/11/12/13/16) share the same round loop shape; if you understand `10_run_fedavg.py` you understand all of them, the only things that change are which client class is instantiated and which aggregation function is called.
- `RUN_NAME` in every script encodes its exact hyperparameters into the output folder/file names (e.g. `dp_fedavg_rounds_50_local_epochs_1_noise_1_clip_1_epscap_8`) — this is how multiple experiment variants coexist under `results/` without overwriting each other.

---

## 11. Federated Learning Methods, Explained

### Centralized (upper baseline)
All 17,084 training records pooled into one `DataLoader`, one model, standard supervised training. Not "federated" at all — it's the ceiling: the best any of the FL methods could hope to match.

### Standard FedAvg
```
w_global = Σ_k (n_k / N) · w_k
```
Every hospital trains 1 local epoch from the current global weights; server averages, weighted purely by dataset size. Simple, this project's non-uncertainty-aware baseline.

### Compatible FedAvg (control)
Identical to Standard FedAvg except the model has been passed through `ModuleValidator.fix()` (the DP-readiness conversion) — with **no** actual DP applied. Purpose: prove any accuracy loss in the DP experiments comes from noise/clipping, not from swapping GroupNorm in.

### Uncertainty-Aware FedAvg (UA-FedAvg) — this project's contribution
```
raw_weight_k   = n_k / (u_k + ε)
final_weight_k = raw_weight_k / Σ_j raw_weight_j
```
where `u_k` is hospital k's MC-dropout predictive-variance score on the **shared validation set** after that round's local training. Confident (low-variance) hospitals get amplified influence; uncertain hospitals get suppressed, independent of how much data they have.

### DP-FedAvg
Same as Standard FedAvg, but each hospital's local training goes through **Opacus DP-SGD**: per-example gradient clipping (`max_grad_norm=1.0`) + calibrated Gaussian noise (`noise_multiplier=1.0`), with Poisson-sampled minibatches so the privacy accountant's math holds. Each hospital accumulates its *own* ε (they differ because each hospital has a different dataset size → different sampling ratio).

### DP-UA-FedAvg
DP-SGD local training **and** uncertainty-aware aggregation combined — the full "both contributions at once" experiment.

---

## 12. Differential Privacy, Explained

**DP-SGD**, per local step:
1. Compute the gradient **for each individual training example** (not just the batch-averaged gradient).
2. **Clip** each per-example gradient to L2 norm ≤ `max_grad_norm` (1.0 here) — bounds any single ECG's maximum possible influence on the update.
3. **Sum** the clipped gradients and add **Gaussian noise** scaled by `noise_multiplier × max_grad_norm` (noise_multiplier=1.0 here).
4. Average and apply the optimizer step as normal.

This project's specific configuration (from the notebook's DP section):

| Setting | Value |
|---|---|
| Noise multiplier | 1.0 |
| Max gradient norm (clipping) | 1.0 |
| Delta (δ) | 1e-5 |
| Privacy accountant | PRV (Opacus `"prv"`) |
| Privacy level | **Record-level** (per-ECG), not per-patient |
| Local epochs / round | 1 |
| Max communication rounds | 50 (early stop patience 5) |
| Max epsilon (reported) | The largest ε accumulated by *any single hospital* at the selected checkpoint (5.4551) |

Because each hospital has a different dataset size (and therefore a different Poisson sampling ratio per round), **ε is hospital-specific** even though noise/clipping settings are identical across hospitals — Hospital 3 (smallest) accumulates ε fastest.

**Why is this "record-level" and not "patient-level" DP?** The privacy unit here is one ECG recording, not one patient. Since some patients contribute multiple ECGs, the formal guarantee is technically weaker than a patient-level bound would be — this is called out explicitly as a limitation (§15).

---

## 13. Final Results (Actual Numbers)

All methods evaluated on the same **2,158-record held-out test split**, fixed decision threshold 0.5, checkpoints selected by validation loss (test labels never touched during model selection).

| Method | Best epoch/round | Test loss | Macro-F1 | Weighted-F1 | Macro-AUROC | ε |
|---|---:|---:|---:|---:|---:|---:|
| Centralized CNN | 10 | 0.3084 | **0.6686** | 0.7216 | **0.8964** | – |
| Standard FedAvg | 15 | 0.3336 | 0.6333 | 0.6900 | 0.8771 | – |
| Compatible FedAvg | 15 | 0.3336 | 0.6333 | 0.6899 | 0.8771 | – |
| UA-FedAvg | 9 | 0.3525 | 0.6134 | 0.6677 | 0.8658 | – |
| DP-FedAvg | 9 | 0.4873 | 0.2750 | 0.3803 | 0.7711 | 5.4551 |
| DP-UA-FedAvg | 8 | 0.4843 | 0.2453 | 0.3524 | 0.7622 | 5.2616 |

### Independent local models (no federation at all)

| Hospital | Best epoch | Test loss | Macro-F1 | Weighted-F1 | Macro-AUROC |
|---|---:|---:|---:|---:|---:|
| Hospital 0 | 8 | 0.3898 | 0.5725 | 0.6344 | 0.8595 |
| Hospital 1 | 15 | 0.3967 | 0.5962 | 0.6487 | 0.8611 |
| Hospital 2 | 14 | 0.3564 | 0.6156 | 0.6677 | 0.8640 |
| Hospital 3 | 17 | 0.5090 | **0.3005** | 0.2841 | 0.8064 |

Standard FedAvg (0.6333) beats **every** individual local model, including the best one (Hospital 2, 0.6156) — the clearest, cleanest demonstration in this project that federation helps.

### DP-FedAvg threshold calibration (validation-only, no retraining, no extra privacy cost)

| Class | Threshold | Fixed F1 (0.5) | Calibrated F1 | Change | AUROC |
|---|---:|---:|---:|---:|---:|
| NORM | 0.45 | 0.7560 | 0.7588 | +0.0028 | 0.8595 |
| MI | 0.21 | 0.3872 | 0.5579 | +0.1707 | 0.7728 |
| STTC | 0.15 | 0.2318 | 0.5276 | +0.2957 | 0.7754 |
| CD | 0.14 | 0.0000 | 0.4890 | **+0.4890** | 0.7365 |
| HYP | 0.09 | 0.0000 | 0.3374 | **+0.3374** | 0.7113 |

Overall: **Macro-F1 0.2750 → 0.5341** (+0.2591), **Weighted-F1 0.3803 → 0.5886** (+0.2083), AUROC unchanged (0.7711 — calibration only moves the decision boundary, not the ranking), exact multi-label subset accuracy 38.88% (839/2,158). Note the fixed 0.5 threshold produced **zero** positive predictions at all for CD and HYP — DP noise had pushed those classes' probabilities uniformly below 0.5, which calibration fixes cheaply.

---

## 14. Discussion — Why the Results Look Like This

1. **Federation worked.** Standard FedAvg (0.6333 Macro-F1 / 0.8771 AUROC) landed close to the centralized ceiling (0.6686 / 0.8964, a −5.3% relative Macro-F1 gap) and beat every hospital training alone.
2. **The DP-compatible architecture itself cost nothing.** Compatible FedAvg reproduced Standard FedAvg to 4 decimal places (0.6333 / 0.8771). So the later DP degradation is **entirely** attributable to clipping + noise, not to swapping BatchNorm-style behavior for GroupNorm.
3. **Raw MC-dropout uncertainty weighting did not help on this partition.** UA-FedAvg (0.6134) sits *below* plain FedAvg (0.6333); DP-UA-FedAvg (0.2453) sits below DP-FedAvg (0.2750). Plausible reason: under this Dirichlet-generated label skew, a hospital's local model can be *confidently* wrong on the shared validation set (e.g. Hospital 3, trained almost exclusively on STTC, may produce low-variance MC-dropout predictions that are still poorly calibrated for the classes it barely saw) — epistemic uncertainty measured via dropout variance doesn't automatically track "how well does this update generalize to the global distribution." This is an honest negative result, not a bug — and it's explicitly one of the project's discussion points.
4. **DP hurt fixed-threshold accuracy a lot.** 0.6333 → 0.2750 Macro-F1 at ε≈5.46, δ=1e-5. Clipping + noise at `noise_multiplier=1.0` is a fairly strong privacy setting for a model this small (38.6k params), so signal-to-noise per update drops sharply.
5. **But DP didn't destroy ranking ability** — AUROC only fell to 0.7711 (vs 0.8771 non-private), even while raw F1 at 0.5 threshold cratered to 0.2750 with two classes at *exactly* zero F1. High AUROC + collapsed F1 is the textbook signature of "the model still ranks positives above negatives reasonably well, but its probabilities are miscalibrated relative to a generic 0.5 cutoff" — which is exactly what motivated §13's threshold calibration step.
6. **Calibration recovered most of the gap for free.** +0.2591 Macro-F1 with zero retraining and zero additional privacy spend — because choosing per-class thresholds only touches the decision rule, not the model or its DP-protected training process. The honest cost: more positive predictions per class → more false positives, which would need per-class clinical risk review before any real deployment.

---

## 15. Limitations

- Results rest on **one dataset, one Dirichlet partition (α=0.5), one primary random seed (42)** — variability across different seeds/partition severities was not measured.
- DP is **record-level, not patient-level** (a patient contributing multiple ECGs doesn't get one combined guarantee).
- Opacus `secure_mode=False` — the DP noise generation is **not** cryptographically secure-random; this implementation is exploratory/research-grade, not deployment-grade privacy.
- Threshold calibration trades false negatives for false positives — clinically this needs per-class risk analysis, not just F1.
- No external clinical validation, no comparison against a cardiologist or an existing clinical PTB-XL benchmark model.
- **The model is not suitable for clinical use** without substantially more validation. (Say this explicitly if asked in the eval — it heads off "so can this diagnose patients?" cleanly.)

---

## 16. How to Run Everything, End to End

### Setup

This repo is `uv`-managed (`pyproject.toml` + `uv.lock`, Python `3.12.x`, CUDA 12.6 PyTorch wheel pinned via `[[tool.uv.index]]`). Two ways to get a working environment:

**Option A — `uv` (matches the lock file exactly):**
```bash
uv sync
```

**Option B — plain pip / venv (see `setup.md`):**
```bash
python -m venv venv
# Windows: .\venv\Scripts\Activate.ps1   |   Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

> **Windows note:** keep the local clone path short (e.g. `D:\uap-fedecg`) — a long path (like this repo currently sits in, under `Final Year\8th Semester\...`) combined with a venv can hit Windows' path-length limit during `pip install`.

### Get the data

Download PTB-XL yourself into `data/raw/ptb-xl/` (it's gitignored — everyone does this locally; there's no bundled downloader script in this checkout despite `setup.md` mentioning one).

### Run in order

```bash
uv run python scripts/01_verify_ptbxl.py             # confirms the dataset download is complete
uv run python scripts/02_prepare_metadata.py          # builds data/processed/*
uv run python scripts/03_smoke_test_dataset.py        # sanity check: Dataset loads correctly
uv run python scripts/04_smoke_test_model.py          # sanity check: model forward/backward works

uv run python scripts/05_train_centralized.py         # centralized baseline (fix the bug in §17 first!)
uv run python scripts/06_evaluate_centralized.py      # centralized test-set metrics

uv run python scripts/07_create_hospital_partitions.py   # build the 4 hospitals
uv run python scripts/08_verify_hospital_partitions.py   # integrity checks
uv run python scripts/08b_inspect_non_iid.py             # confirm the skew is "enough"

uv run python scripts/09_train_local_baselines.py     # per-hospital, no federation

uv run python scripts/10_run_fedavg.py                # Standard FedAvg
uv run python scripts/11_run_uncertainty_fedavg.py    # UA-FedAvg
uv run python scripts/13_run_compatible_fedavg.py     # architecture-only control
uv run python scripts/12_run_dp_fedavg.py             # DP-FedAvg
uv run python scripts/16_run_dp_ua_fedavg.py --max-rounds 50 --mc-passes 10   # DP-UA-FedAvg

uv run python scripts/17_finalize_project_results.py  # combine everything into final_model_comparison.csv

uv run python scripts/19_calibrate_class_thresholds.py \
    --checkpoint results/checkpoints/dp_fedavg_.../dp_fedavg_best.pt --method-name DP-FedAvg
uv run python scripts/18_show_model_predictions.py    # example predictions from the calibrated checkpoint
```

Each of the FedAvg-family scripts (10/11/12/13/16) takes a while — 50 rounds × 4 hospitals × 1 local epoch each, plus (for DP scripts) Opacus's per-sample gradient overhead. Budget real time for these, especially DP ones, on CPU.

---

## 17. Known Issues / Bugs to Check Before Demoing

**`scripts/05_train_centralized.py` will currently crash immediately (`NameError`) as committed.** The last commit touching this file (`f59eb2c`, "Complete standard FedAvg baseline with early stopping") removed the top-of-file `from pathlib import Path` import while leaving a `Path(...)` usage a few lines *above* the (now later) `from pathlib import Path` import, and separately renamed the constant `FULL_EPOCHS` → `MAX_EPOCHS` but left one leftover reference to the old, now-undefined name `FULL_EPOCHS` at the point where it picks the epoch count for a full (non-debug) run.

Concretely, two independent fixes are needed:
1. Move `from pathlib import Path` above the first use of `Path(__file__)` (line ~18), or add a second early import.
2. Replace the stray `FULL_EPOCHS` reference (`number_of_epochs = FULL_EPOCHS`) with `MAX_EPOCHS`.

Everything else read through this repo (scripts 01–04, 06–19, all of `src/`) is internally consistent and free of this class of bug — this appears to be an isolated slip from that one commit. Worth fixing before you need to demo a live centralized-training run tomorrow; ask if you'd like it patched now.

---

## 18. Architectural Decisions — Q&A

**Q: `flwr` (Flower) is a core dependency — why doesn't any script actually import it?**
A: It doesn't. Every experiment script (`scripts/10`–`16`) implements the FL round loop, client `.fit()` calls, and aggregation entirely with plain Python/PyTorch/NumPy (`src/federated/*.py`), and `server_app.py` is a deliberate empty stub. This gives the team full, transparent control over exactly how aggregation happens (critical for iterating on the novel uncertainty-weighted aggregation formula) and avoids Flower's networking/simulation abstraction layer getting in the way of tight experiment loops, logging, and Opacus integration. `flwr` in `pyproject.toml` most likely reflects the originally-planned approach before the team moved to a manual loop.

**Q: Why a small 1D-CNN (38.6k params) instead of something bigger (ResNet-1D, Transformer, LSTM)?**
A: FL simulation multiplies cost by (number of clients × number of rounds), and DP-SGD multiplies it again (per-example gradients, not batched). A compact model keeps ~50-round × 4-hospital × 2 DP variants experiments tractable on a single laptop GPU (an RTX 3050 6GB, per the notebook's environment section), while still being expressive enough for 5-class ECG classification (macro-AUROC 0.90 centralized).

**Q: Why GroupNorm instead of BatchNorm?**
A: BatchNorm's running mean/variance statistics are computed across a batch and don't have a well-defined per-example gradient — incompatible with DP-SGD's per-example clipping, and Opacus's `ModuleValidator` will refuse to privatize a model containing it. GroupNorm normalizes within each individual sample, so exactly one architecture works for centralized, FedAvg, *and* DP training without special-casing.

**Q: Why multi-label (5 independent sigmoids) instead of single-label softmax?**
A: PTB-XL ECGs frequently satisfy more than one diagnostic superclass simultaneously (e.g. MI + STTC). Forcing softmax would require picking one "true" label per ECG, discarding real co-occurrence information the dataset actually encodes.

**Q: Why Dirichlet partitioning instead of some other non-IID scheme?**
A: Dirichlet-based label-skew partitioning (concentration α) is the standard, widely-used method in FL research for generating controllable, reproducible non-IID splits — lower α = more skew. It's also patient-safe here (whole patients move together, never split ECGs from one patient across hospitals) and validated by two dedicated scripts (08, 08b) rather than just assumed correct.

**Q: Why MC-Dropout for uncertainty, rather than a Bayesian NN or a deep ensemble?**
A: MC-Dropout needs no architectural changes (the model already has dropout for regularization) and no extra training cost (uncertainty is estimated by running N stochastic forward passes at inference time) — cheap enough to run every round, for every hospital, without materially slowing the FL loop down. Deep ensembles would multiply per-hospital training cost by the ensemble size; a full Bayesian NN would require redesigning the layers.

**Q: Why is `results/` mostly gitignored?**
A: Model checkpoints, full training logs, and figures are large, machine-specific (paths, GPU), and fully reproducible from the scripts + a fixed seed — so only a curated set of final outputs actually needed by the report notebook (the DP-FedAvg calibration folder, a few example predictions) is committed; everything else regenerates locally.

**Q: What's `main.py` for?**
A: Nothing functional — it's the default `uv init` placeholder ("Hello from ..."). The real entry points are the numbered `scripts/`.

---

## 19. Evaluation-Day Cheat Sheet

Fast answers to the questions most likely to come up.

- **"What's your baseline and what's your best number?"** Centralized CNN is the ceiling: 0.6686 Macro-F1 / 0.8964 AUROC. Standard FedAvg gets to 0.6333 / 0.8771 — within ~5% of centralized while keeping data local.
- **"Did your uncertainty idea work?"** Be honest: no, not on this partition — UA-FedAvg (0.6134) and DP-UA-FedAvg (0.2453) both underperformed their sample-count-only counterparts. That's a real, reported, discussed finding (§14, point 3), not a hidden failure.
- **"What did privacy cost you?"** Macro-F1 dropped from 0.6333 (non-private) to 0.2750 (DP-FedAvg) at ε≈5.46, δ=1e-5 — but AUROC only dropped to 0.7711, and post-hoc calibration recovered Macro-F1 to 0.5341 for free (no retraining, no extra ε).
- **"Is GroupNorm/architecture the reason DP hurt so much?"** No — proven by the Compatible-FedAvg control, which matches Standard FedAvg exactly (0.6333/0.8771). The drop is 100% attributable to DP-SGD's clipping+noise.
- **"How did you make it non-IID?"** Dirichlet(α=0.5) sampling of *patients* per primary diagnostic class, across 4 hospitals, seed 42 — verified by dedicated integrity + skew-sufficiency scripts.
- **"Is this deployable / clinically usable?"** No — say so directly. One dataset, one seed, record-level (not patient-level) DP, no external clinical validation (§15).
- **"Why not use Flower since it's in your dependencies?"** Manual loop gave full control for developing/debugging the novel uncertainty-aggregation logic and DP integration together; see §18.
- **"What library did the actual privacy accounting?"** Opacus, `PrivacyEngine`, `"prv"` accountant, reporting the max ε across hospitals at the selected checkpoint.

---

## 20. Notebook Guide

`notebooks/Final_ECG_FL_Project_Notebook.ipynb` (also exported as `.html`) is the actual final report — it reads already-completed results from `results/` rather than recomputing anything live, so it's fast to open and walk through in the room. Its 14 sections, in order: **Abstract → Introduction/Objectives → Experimental Environment → Dataset/EDA → Preprocessing & Non-IID Hospitals → Model Architecture → Methods → DP Configuration → Final Fixed-Threshold Results → DP-FedAvg Threshold Calibration → Example Predictions → Discussion → Limitations → Conclusion → References.** Every number quoted in §13/§14/§15 above is pulled directly from this notebook, so if you present from the notebook and get asked to cross-check a number against "the README", they'll match.

`01_data_exploration.ipynb` and `final_project_notebook.ipynb` are empty starter notebooks — not part of the final deliverable.

---

## 21. Tech Stack

| Tool | Role |
|---|---|
| PyTorch | Model definition, training, autograd |
| Opacus | DP-SGD (per-example gradients, clipping, noise, privacy accounting) |
| wfdb | Reading PTB-XL's WFDB-format waveform files |
| pandas / NumPy | Metadata handling, partitioning, parameter aggregation math |
| scikit-learn | F1, AUROC, threshold-sweep metrics |
| matplotlib | All figures (training curves, ROC, confusion matrices, calibration plots) |
| flwr (Flower) | Listed dependency, **not actually used** — see §18 |
| uv | Dependency/environment management (`pyproject.toml` + `uv.lock`) |

---

## 22. References

1. P. Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," *Scientific Data*, vol. 7, art. 154, 2020.
2. C. Dwork, F. McSherry, K. Nissim, A. Smith, "Calibrating noise to sensitivity in private data analysis," *TCC*, LNCS vol. 3876, Springer, 2006.
3. T. Li, A. K. Sahu, A. Talwalkar, V. Smith, "Federated learning: Challenges, methods, and future directions," *IEEE Signal Processing Magazine*, vol. 37, no. 3, 2020.
