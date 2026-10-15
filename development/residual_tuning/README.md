# Figure 5 High-OCN Residual Tuning Package

This package runs a small, strictly nested residual-model search for the Figure 5 F2-P8 base model. It is designed for local execution from any directory after unzipping. The expensive nested tuning was not run during package preparation.

The packaged input is the 917-pair training-only table. It contains no 1,794-pair held-out target data. No files outside this package are required to run the tuning analysis.

## Setup

Option A, venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Option B, conda:

```bash
conda env create -f environment.yml
conda activate fig5_residual
```

On Apple Silicon, if regular TensorFlow installation is slow or unavailable, install the current Apple-supported TensorFlow stack in the environment following Apple's TensorFlow plugin guidance, then rerun the dry run. Keep the package configs unchanged.

## Validate Package

```bash
python scripts/run_nested_residual_tuning.py \
  --input inputs/fig5_917_F2_P8.csv \
  --config config/run_config.json \
  --grid config/residual_grid.json \
  --base-config config/base_model.json \
  --output outputs \
  --dry-run
```

The dry run loads the input, checks columns and row counts, constructs deterministic folds, enumerates candidates, estimates work, checks output writability, and exits before model training.

## Run

```bash
/path/to/residual_tuning/run_all.sh
```

`run_all.sh` changes into the package directory before running, so it can be called from any current working directory.

Equivalent explicit command:

```bash
cd /path/to/residual_tuning
python scripts/run_nested_residual_tuning.py \
  --input inputs/fig5_917_F2_P8.csv \
  --config config/run_config.json \
  --grid config/residual_grid.json \
  --base-config config/base_model.json \
  --output outputs \
  --resume
```

Then run:

```bash
python scripts/evaluate_tail.py --predictions outputs/outer_oof_predictions.csv --output outputs
python scripts/bootstrap_tail.py --predictions outputs/outer_oof_predictions.csv --output outputs --n-resamples 5000
python scripts/make_figures.py --output outputs
python scripts/validate_outputs.py --output outputs
```

## Resume

```bash
python scripts/run_nested_residual_tuning.py \
  --input inputs/fig5_917_F2_P8.csv \
  --config config/run_config.json \
  --grid config/residual_grid.json \
  --base-config config/base_model.json \
  --output outputs \
  --resume
```

The tuner writes one checkpoint per completed outer fold in `outputs/checkpoints/`. With `--resume`, completed folds are skipped.

## Workload

Outer CV has 5 seeds x 5 folds = 25 outer folds.

The residual grid has 4 Ridge candidates and 108 TensorFlow NN candidates, for 112 residual candidates per outer fold. The full tuning run therefore trains 2,800 residual candidates.

For each outer fold, the package records three compatible residual comparisons when available: the selected tuned residual model, the best Ridge residual benchmark, and the historical fixed NN configuration `(16, 8), l2=5e-4, learning_rate=2e-4, Huber delta=0.75`.

The fixed base model is an ensemble of 5 MLPs. Each outer fold fits one outer base ensemble and 5 inner-CV base ensembles to create inner-OOF residual targets, for 750 base MLP fits total.

Default TensorFlow parallelism is `n_jobs = 1`. You may increase `parallelism.sklearn_n_jobs` for the sklearn base ensembles if your machine has enough CPU cores. Avoid running multiple TensorFlow model trainings simultaneously unless you know your GPU/CPU memory limits.

## Outputs

Main tuning outputs:

- `outputs/inner_tuning_results.csv`
- `outputs/outer_fold_results.csv`
- `outputs/outer_oof_predictions.csv`
- `outputs/residual_fit_calibration_ids.csv`
- `outputs/tail_metrics.csv`
- `outputs/high_OCN_pair_improvements.csv`
- `outputs/tail_bootstrap_differences.csv`
- `outputs/high_OCN_residual_vs_OCN_<residual_model_label>.pdf`
- `outputs/high_OCN_absolute_error_<residual_model_label>.pdf`

Selection is based on high-OCN RMSE, with high-OCN MAE as the secondary tie-breaker and simpler models preferred after ties. Global R2 is not used for residual-model selection.
