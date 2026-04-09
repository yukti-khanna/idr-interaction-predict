# idr-interaction-predict

Prediction-focused workflow for ranking pairwise interactions between intrinsically disordered regions (IDRs) using sequence-derived features and a calibrated hybrid MLP model.

This repository is the prediction-side companion to `idr-interaction-analysis`.

---

## What this repository does

This repo currently supports two workflows.

### 1) Local feature workflow
Starting from a FASTA file plus either:
- an explicit pair list, or
- two ID lists,

it will:
- generate canonical pair definitions
- compute pairwise sequence features

This is the **default** and currently most portable workflow.

### 2) Pretrained prediction workflow
Starting from a computed feature table, it will:
- load the pretrained calibrated hybrid model bundle
- predict interaction strength
- rank pairs by predicted score

This workflow is available, but the shipped pretrained bundle is still **environment-sensitive** and is currently tested in the original compatible prediction environment.

---

## Repository structure

```text
idr-interaction-predict/
├── README.md
├── Snakefile
├── environment-features.yml
├── environment-predict.yml
├── config/
│   └── config.yaml
├── inputs/
│   ├── fasta/
│   │   ├── sample.fasta
│   │   └── validated_final.fasta
│   └── lists/
│       ├── sample_pairs.txt
│       ├── idr_a.txt
│       └── idr_b.txt
├── models/
│   └── pretrained/
├── resources/
│   └── residues.csv
├── results/
│   ├── prepared/
│   ├── features/
│   └── predictions/
└── scripts/
    ├── generate_pairs.py
    ├── compute_pair_features.py
    ├── predict_with_hybrid_model.py
    └── mlp_with_plateau_residual_calibrated.py
```

---

## Environments

Two environments are provided because the portable part of the repo and the pretrained prediction step are currently best handled separately.

### `environment-features.yml`
Use this environment for:
- pair generation
- feature computation

Create it with:

```bash
conda env create -f environment-features.yml
conda activate idr_features
```

### `environment-predict.yml`
Use this environment for:
- pretrained prediction with the shipped calibrated bundle

This file was exported from the compatible environment used for prediction.

Create it with:

```bash
conda env create -f environment-predict.yml
conda activate <env-name-from-yml>
```

---

## Quick start

### Default local workflow
From the repository root:

```bash
snakemake --cores 1
```

This builds:
- `results/prepared/pairs.csv`
- `results/features/pair_features.csv`

The default target intentionally stops at feature generation.

### Pretrained prediction
Run prediction explicitly:

```bash
snakemake --cores 1 predict_pretrained
```

This writes:
- `results/predictions/pair_predictions.csv`
- `results/predictions/ranked_pairs.csv`

---

## Manual commands

### Generate pairs from an explicit pair list

```bash
python scripts/generate_pairs.py \
  --fasta inputs/fasta/sample.fasta \
  --pairs-file inputs/lists/sample_pairs.txt \
  --keep-order \
  --output results/prepared/pairs.csv
```

### Compute pairwise features

```bash
python scripts/compute_pair_features.py \
  --pairs-csv results/prepared/pairs.csv \
  --fasta inputs/fasta/sample.fasta \
  --residues-file resources/residues.csv \
  --output-csv results/features/pair_features.csv
```

### Predict with the pretrained bundle

```bash
python scripts/predict_with_hybrid_model.py \
  --features-csv results/features/pair_features.csv \
  --bundle models/pretrained/mlp_plateau_calibrated_bundle_sel_cols.pkl \
  --output-csv results/predictions/pair_predictions.csv \
  --ranked-output-csv results/predictions/ranked_pairs.csv
```

---

## Configuration

The current `config/config.yaml` uses:

- FASTA: `inputs/fasta/sample.fasta`
- explicit pairs: `inputs/lists/sample_pairs.txt`
- optional list-A input: `inputs/lists/idr_a.txt`
- optional list-B input: `inputs/lists/idr_b.txt`
- residues table: `resources/residues.csv`
- pretrained model bundle: `models/pretrained/mlp_plateau_calibrated_bundle_sel_cols.pkl`

---

## Pair input modes

`generate_pairs.py` supports two ways to define candidate interactions.

### Explicit pair list
Example format:

```text
idr_1__idr_2
idr_3__idr_4
idr_2__idr_4
```

### Two-list mode
Provide:
- list A IDs
- list B IDs

and generate all allowed combinations.

To switch modes, edit `config/config.yaml`:

```yaml
pairing:
  mode: two_lists
```

---

## Outputs

### Feature workflow
- `results/prepared/pairs.csv`
- `results/features/pair_features.csv`

### Prediction workflow
- `results/predictions/pair_predictions.csv`
- `results/predictions/ranked_pairs.csv`

---

## Current status

### Stable now
- canonical pair generation
- local feature computation
- clean local features environment
- separate prediction target in Snakemake

### Still environment-sensitive
- loading the shipped pretrained bundle across all local platforms and package combinations

---

## Recommended use right now

### Use locally for
- generating canonical pair sets
- computing sequence-derived pair features

### Use the compatible prediction environment for
- loading the shipped pretrained bundle
- producing ranked predictions

If pretrained inference fails locally, use `environment-predict.yml`.

---

## Future improvement

A more portable future release should replace the single monolithic bundle with separate artifacts such as:
- `base_mlp.pkl`
- `scaler.pkl`
- `residual_nn.keras`
- `model_metadata.json`

That would make pretrained inference easier to run across environments.

This project is licensed under the MIT License - see the LICENSE file for details.
