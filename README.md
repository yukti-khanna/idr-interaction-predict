# rAHD–RG/RGG interaction predictor

Predict interaction strength between an rAHD sequence and an RG/RGG sequence using the final 13-descriptor hybrid model. The output is predicted **log₁₀(−B₂₂)**.

The workflow starts from amino-acid sequences. It calculates descriptors, applies a frozen five-member MLP ensemble, then adds the stored gated and bounded residual correction. It does not train or recalibrate on user inputs.

## Quick start: two stages with Snakemake

Run commands from the repository root. Existing examples are in `inputs/`.

1. Activate your feature environment (NumPy, pandas, SciPy; Numba is optional) and run:

```bash
snakemake --cores 4 prepare_features
```

2. Activate the environment used for the authoritative frozen model, which includes TensorFlow, joblib and scikit-learn, and run:

```bash
snakemake --cores 4 predict_final
```

Snakemake must be available in the environment running each command. The workflow uses the active environment; it does not silently activate or install one. `requirements-features.txt` and `requirements-inference.txt` list the Python requirements. The environment YAMLs record earlier environments; their platform-specific packages are not universal installation recipes.

Feature preparation does not load TensorFlow. Prediction reuses the saved feature table. `snakemake --cores 4` is an alias for preparing inputs/features; `predict_pretrained` remains a compatibility alias for `predict_final`.

**Deployment status:** the authoritative frozen model is installed; all copied artifact checksums match. See [server environment instructions](environments/README.md). Runtime identity must pass in the inference environment.

## Sequence inputs

Edit `config/config.yaml` to choose a FASTA and a plain-text pair list. Existing example files have not been changed.

```text
RGG_identifier__rAHD_identifier
another_RGG_identifier__another_rAHD_identifier
```

There is one pair per line, no header. Identifiers must match FASTA headers. Both sequence types can be in the same FASTA. **RG/RGG is first, rAHD second**; keep `pairing.keep_order: true`. The model is role-specific, so partners must not be sorted alphabetically. The two-list mode uses list A for RG/RGG and list B for rAHD.

Sequences require standard amino-acid letters and at least five residues. Users do not supply features, RDFs, simulation values or B₂₂ targets.

## Install the authoritative frozen model

Copy the unchanged server model directory to an accessible location, then run:

```bash
python scripts/install_frozen_model.py --source /path/to/server/frozen_model
```

The installer verifies all four artifact checksums against `config/accepted_model_hashes.json` **before copying** into `models/final_frozen/`. The prediction loader repeats artifact checks and runs known-reference numerical identity checks. A mismatch stops execution. No weights, scalers, feature order, lambda, gate or cap are inferred or altered.

The required artifacts are `MODEL_MANIFEST.json`, `base_model_ensemble.joblib`, `residual_preprocessor.joblib` and `residual_nn.keras`. `models/legacy/` contains the older pretrained models; they are not used by the active workflow.

## Outputs

The Snakemake paths are configured in `config/config.yaml`:

- `results/prepared/pairs.csv`: validated pair identifiers.
- `results/features/pair_features.csv`: sequences and final model descriptors.
- `results/predictions/pair_predictions.csv`: base, raw residual, gate, applied correction and final prediction, plus five individual base predictions.
- `results/predictions/ranked_pairs.csv`: predictions sorted by final score, descending.
- Identity-check CSV/text files beside feature and prediction outputs.

Predictions remain on the log₁₀(−B₂₂) scale; they are not probabilities. A prediction is produced from sequences and does not establish whether an independently simulated B₂₂ is negative. Range warnings in the extended CLI are descriptive, not automatic exclusions.

## Extended CLI and prepared-run reuse

The same implementation is also available without Snakemake:

```bash
python scripts/predict_pairs.py --pairs inputs/lists/sample_pairs.txt \
  --rahd-reference inputs/fasta/sample.fasta \
  --rgg-reference inputs/fasta/sample.fasta \
  --features-only --run-name sample_features

python scripts/predict_pairs.py \
  --prepared-run results/runs/sample_features --run-name sample_predictions
```

Keep the entire preparation directory, not just one CSV. Each run gets a new directory under `results/runs/`; existing runs are not overwritten. `--identity-only` checks the frozen model on known references without predicting user pairs. `--rahd-fasta` and `--rgg-fasta` with `--cross-product` support FASTA cross products. Historical `--ahd-*` aliases remain supported.

## What the descriptors mean

The five global descriptors are −NCP (negative product of the partners’ net charges), mean residue lambda for each partner, and SHD for each partner. Lambda describes hydropathy/stickiness; SHD describes its sequence patterning.

The eight local descriptors summarize variation across overlapping **five-residue sequence windows**: local charge complementarity, SCD for each partner, RG content on the RG/RGG side, aromatic content on the rAHD side, SHD for each partner, and Ashbaugh–Hatch interaction propensity. Historical code calls these windows “patches.” They are sequence windows, not structural surface patches. Patch1 is RG/RGG and patch2 is rAHD; window step is one residue and standard deviations use `ddof=0`.

The validated representation uses two fixed parameter tables: `resources/global_residues.csv` for global mean lambda/SHD, and `resources/residues.csv` for local descriptors. They are intentionally distinct. Mean lambda is the sequence sum divided by sequence length. Do not interchange the tables or replace means with sums.

`config/final_feature_schema.json` records exact order and aliases. Historical internal AHD/TAD identifiers remain for serialized-model compatibility; the scientific name used here is rAHD.

## Final predictor

The base is the mean of five frozen MLP predictions. Using the parameters stored in the manifest:

```text
correction = lambda × sigmoid((−NCP − centre) / scale)
             × clip(raw residual, −cap, cap)
final prediction = base prediction + correction
```

The cap acts on the raw residual before gating and weighting. Training used only the 917-pair development set. Stored evaluation results are in `results/reference_predictions/`; they are reference outputs, not inputs to user prediction.

## Repository map

| Folder | Contents |
|---|---|
| `inputs/` | Existing user-editable sequence examples and pair lists |
| `scripts/` | Pair preparation, descriptors, frozen inference and checks |
| `scripts/vendor/` | Preserved historical scientific function bodies |
| `resources/` | Fixed residue parameters and small identity/range reference tables |
| `config/` | Input/output paths, feature schema and accepted model hashes |
| `models/final_frozen/` | Authoritative deployed model artifacts |
| `models/legacy/` | Older model files, excluded from active inference |
| `results/` | User runs, validation checks and saved reference predictions |
| `development/` | Descriptor-selection/residual-development history and legacy code/results |
| `tests/` | Input, alignment, feature-identity and inference-guard tests |
| `docs/` | Model history, integration notes and file-move record |

`development/patches/` retains the historical local-feature training table. The active workflow computes these descriptors directly from sequences. Development history is separate from deployment: the archived F2 script references an original CV-fold file not yet recovered, so a complete from-scratch reconstruction of selection is **not yet certified**. Do not generate different folds and claim they reproduce the historical selection.

Older manuscript control exports have a recorded global-parameter convention discrepancy. They are not bundled here or silently treated as sequence-to-prediction identity references. The active identity references are from the accepted frozen development predictions. Control figures remain in the separate analysis workspace.

## Validation

```bash
python -m unittest discover -s tests/unit -v
```

The model-inference test is opt-in and requires the authoritative artifacts/environment:

```bash
RUN_FROZEN_MODEL_IDENTITY=1 python -m unittest discover -s tests/unit -v
```

No new-data result changes model parameters. See `docs/model_history.md` for unresolved historical issues and `docs/file_moves.json` for the reorganisation record.

## Train on your own data

A separate custom training workflow accepts sequences, explicit target definitions and fixed train/validation/test splits. Base and optional hybrid modes save a new model directory; they never overwrite the supplied frozen predictor. See [custom training](docs/CUSTOM_TRAINING.md) for commands, calibration details and validation limitations.
