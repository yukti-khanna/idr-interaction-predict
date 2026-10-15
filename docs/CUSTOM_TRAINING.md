# Train a model on your own data

Custom training is separate from deployment of the supplied frozen model. It never overwrites `models/final_frozen/`, and does not reproduce the manuscript's full descriptor-selection or nested-tuning study. It reuses the fixed 13-descriptor representation and provides a new, explicit training procedure.

## Data

Supply a CSV with these columns:

```csv
pair_id,rAHD_sequence,RGG_sequence,target,split
pair_001,DEDEDEDEDEDEDEDE,RGRGRGRGRGRGRGRG,3.1,train
```

This single line illustrates the schema, not a sufficient training dataset. Provide at least 10 training rows and 3 validation rows; substantially larger, representative datasets are needed for meaningful results. `split` is `train`, `validation`, or optional `test`. Sequences require >=5 standard residues. Pair IDs and sequence pairs must be unique. Consolidate simulation/experimental replicas upstream and retain their uncertainty separately.

Choose splits before training. By default, no sequence may occur in more than one split, including across partner roles. `--split-unit pair` explicitly permits shared sequences and records overlap counts; pair-disjoint results must not be described as sequence-exclusive generalization.

Targets are used exactly as supplied: no B22 correction, logarithm, sign change or unit conversion is applied. State units, correction and transformation in `--target-description`. For log10(-B22), define it only when B22<0; do not invent values for positive B22.

## Base ensemble

```bash
python scripts/train_custom_model.py \
  --data inputs/custom_training.csv \
  --output models/custom/my_model_v1 \
  --target-column target \
  --target-description 'log10(-corrected B22); specify the B22 units and correction here' \
  --mode base
```

The scaler and five MLP members are fitted only on training rows. Architecture and seeds match the deployed base framework: hidden layers 128/64/32/16, tanh, alpha .01, Adam, initial learning rate .01, max_iter 500, seeds 0–4. Convergence warnings should be inspected. Features use the validated global/local residue tables.

## Optional hybrid training

Use `--mode hybrid` in the TensorFlow environment. This is appropriate only if the dataset contains enough high-−NCP observations and the target scale makes the fixed residual architecture/loss appropriate. At least 10 training pairs with −NCP>=40 are required; that threshold is a computational minimum, not evidence of sufficient data.

- Five-fold shuffled CV (seed42), entirely within TRAIN, generates base OOF residual targets. Each fold fits its own scaler and five base models.
- A residual scaler/network is fitted only on high-−NCP training rows, using these OOF residuals and sigmoid sample weights.
- Residual architecture: Dense16/8 tanh, L2 .0005; Adam .0002; Huber delta .75; fixed 400 epochs by default, seed20260916. `--residual-epochs` is a prespecified user setting, not selected on test performance.
- Gate centre40 and scale12 are fixed. Cap is the 85th percentile of absolute TRAIN OOF residuals among high-−NCP rows.
- Lambda is bounded to [0,1] and fitted by least squares on VALIDATION predictions only. No test target is used. This is a custom calibration procedure, not a claim to replicate the historical nested model-development pipeline.
- The trained model and calibration are saved before optional TEST metrics are calculated.

Training metrics are in-sample. Hybrid validation metrics are calibration-set metrics, not independent validation. Only a genuinely untouched test set gives independent evaluation. Do not repeatedly tune decisions against its reported metrics.

## Snakemake

Edit `config/custom_training.yaml`, then run explicitly:

```bash
snakemake train_custom --cores 4 --configfile config/custom_training.yaml
```

Default feature/inference targets never trigger training. Training is serial in this first implementation; requesting more cores does not parallelize the ensemble fits.

## Use a custom model

```bash
python scripts/predict_custom_model.py \
  --model models/custom/my_model_v1 \
  --data inputs/new_sequence_pairs.csv \
  --output results/custom/my_model_v1_predictions.csv
```

New data require `pair_id,rAHD_sequence,RGG_sequence`; no targets/splits are needed. Predictions are on the target scale documented in the custom manifest. Only load your own trusted model artifacts. Custom inference verifies saved checksums, feature order, adapter hash and residue/implementation hashes.

Each new output directory contains the ensemble/scaler, optional residual model, target definition, input hash, split counts/overlap, software versions, calibration, predictions and metrics. Existing directories are never overwritten. For a rerun use a new directory. Models failing mid-training are incomplete and must not be deployed.

## Validation status

The base-mode execution and custom prediction round trip are tested on explicitly synthetic data. Hybrid TensorFlow training requires execution in the supplied server environment and is not yet runtime-validated locally. These checks do not establish predictive performance on a biological dataset.
