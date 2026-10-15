#!/bin/bash
#SBATCH --job-name=run_2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=10G
#SBATCH -t 24:00:00
#SBATCH -o logs/run_1.out
#SBATCH -e logs/run_1.err



# Run your Python script
python scripts/run_nested_residual_tuning.py \
  --input inputs/fig5_917_F2_P8.csv \
  --config config/run_config.json \
  --grid config/residual_grid.json \
  --base-config config/base_model.json \
  --output outputs/ \
  --resume

