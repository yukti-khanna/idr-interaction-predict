#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/run_nested_residual_tuning.py \
  --input inputs/fig5_917_F2_P8.csv \
  --config config/run_config.json \
  --grid config/residual_grid.json \
  --base-config config/base_model.json \
  --output outputs \
  --resume

python scripts/evaluate_tail.py --predictions outputs/outer_oof_predictions.csv --output outputs
python scripts/bootstrap_tail.py --predictions outputs/outer_oof_predictions.csv --output outputs --n-resamples 5000
python scripts/make_figures.py --output outputs
python scripts/validate_outputs.py --output outputs
