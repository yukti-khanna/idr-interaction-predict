# Final AHD–RG/RGG inference pipeline

Prepared for the mirrored `fig5` clone on the server. Copy this **prediction_pipeline/** directory into that clone. Keep the existing authoritative server model at `outputs/final_hybrid_model/frozen_model/`. There is no MODEL_DIR to edit and no local model artifact is bundled or substituted.

All package, model, resource and output locations are resolved from the script location. Input paths supplied on the command line are relative to your current working directory, or may be absolute. Existing runs are never overwritten.

## Preparation status and important provenance findings

- All 13 regenerated descriptors passed identity on 20 frozen development references: 10 ordinary rows and 10 high-negative-net-charge-product rows with nonzero correction. Maximum absolute difference: **4.6629367034256575e-15**. No held-out rows were used.
- Seven local tests passed; one server model-inference test was intentionally skipped. No local Keras artifact was opened or run.
- **Prediction identity is pending on the server.** Every inference run checks the accepted manifest/artifact hashes and reruns feature/prediction identity before any new predictions.
- Expected accepted manifest SHA256: `48b7279b7e76260c04c1f42ee766d3f385f3d27c5ad61b99b2941d0d02b0d516`.
- The manifest currently in the local workspace is different (`fdb9ad1c7d2e0cc80edbac563960957b553c16aff46780656d047d30d7b7a6e9`). It was neither edited nor substituted. The server must contain the accepted artifact hashes in `config/accepted_model_hashes.json`. A mismatch stops inference with an explicit error; do not overwrite the server model with the local copy.

### Historical global/patch parameter split

The frozen 917-row model input was constructed by joining authoritative global descriptors to historical patch descriptors. The two sources use different fixed residue-lambda tables:

- Global mean lambda and global SHD use the original table, bundled unchanged as `resources/global_residues.csv` (SHA256 `2cf55bc77a7ff8a25b7cc841ac947ff5c6ec33a8fdc3a5e575044152c7d0bbc7`). This table was read from the original CALVADOS resources and copied into this package for portability. Its absolute read-only source is recorded in `resources/SOURCE_PROVENANCE.json`.
- Patch SHD and local AH features use the historical patch table, bundled unchanged as `resources/residues.csv`. Charge columns agree between these tables.
- The same historical function bodies run in separate namespaces with the corresponding table. Mean lambda is always the **sequence sum divided by length**.

Using the patch table for all 13 inputs fails training feature identity (mean-lambda discrepancy up to 0.04256 and global-SHD discrepancy up to 0.22041 in the 20 references). The final adapter reproduces the actual trained representation, without fitting or estimating any residue parameter.

**Older accepted S8 control exports have a separate preprocessing inconsistency:** their four global hydropathy descriptors were generated with the patch table. A 10-row read-only comparison is retained in `validation/historical_S8_feature_convention_audit.csv`. Those exported rows are not interchangeable with the training-consistent sequence representation. This package does not modify or regenerate those controls or figures, and does not claim their sequence-to-prediction identity. The mandatory regression reference is the accepted *in-sample frozen development prediction*, not OOF predictions. Do not mix future training-consistent outputs with old S8 exports without addressing this provenance difference explicitly.

## First command on the server

Activate the existing environment that successfully ran the accepted final model, then, from the server's `fig5` root:

```bash
python prediction_pipeline/predict_pairs.py --identity-only --run-name server_identity
```

Review `prediction_pipeline/prediction_runs/server_identity/pipeline_identity_test.txt`. All 13 features and base/raw residual/gate/applied correction/final prediction must pass. A subsequent inference run also performs this check; there is no skip-identity option.

## Your input files: plain text pairs plus FASTAs

Place your files anywhere on the server and supply their paths in the command. For example, create an `inputs/` directory under the server's `fig5` root containing:

```text
fig5/
    inputs/
        pair_names.txt
        ahds.fasta
        rggs.fasta
    prediction_pipeline/
```

`pair_names.txt` is a plain text file with **one pair per line and no header**:

```text
RGG1__AHD1
RGG2__AHD2
```

The order is **RG/RGG first, AHD second**, separated by two underscores. `RGG1` must match a header in `rggs.fasta`, and `AHD1` must match a header in `ahds.fasta`. The pair-name file contains identifiers only; the FASTAs provide the sequences. No CSV conversion is necessary.

Run from the server's `fig5` root:

```bash
python prediction_pipeline/predict_pairs.py \
  --pairs inputs/pair_names.txt \
  --ahd-reference inputs/ahds.fasta \
  --rgg-reference inputs/rggs.fasta \
  --run-name experiment_01
```

These filenames are examples, not required defaults. Change each command-line path to your actual filename. Paths can also be absolute. `--run-name` names the output directory, which for this example is `prediction_pipeline/prediction_runs/experiment_01/`; it does not select an input file. Use a new run name for each run.

The shell-wrapper equivalent is:

```bash
bash prediction_pipeline/run_predictions.sh \
  --pairs inputs/pair_names.txt \
  --ahd-reference inputs/ahds.fasta \
  --rgg-reference inputs/rggs.fasta \
  --run-name experiment_01
```

The wrapper does not activate an environment. An absolute path to either executable works from any directory; relative input paths are interpreted from the directory where you run the command.

## Supported input modes and tiny examples

All examples are bundled and use real reference sequences. Replace the example file paths with your own inputs. Commands below are relative to the server's `fig5` root.

**A. Explicit pairs with sequences** (`examples/pairs.tsv`):

```bash
python prediction_pipeline/predict_pairs.py \
  --pairs prediction_pipeline/examples/pairs.tsv --run-name example_explicit
```

Columns: `pair_id` (optional), `AHD_name`, `RGG_name`, `AHD_sequence`, `RGG_sequence`. CSV and TSV are supported. Explicit pair IDs are preserved.

**Plain text pair-name file** (`examples/pair_names.txt`, no header):

```text
RGG_name__AHD_name
another_RGG_name__another_AHD_name
```

Use the exact FASTA identifiers, RG/RGG first and AHD second. Blank lines and lines beginning with `#` are ignored. Each pair must have exactly one `__` separator; malformed or unresolved pairs are retained in `rejected_pairs.csv`. For partner identifiers that themselves contain `__`, use the explicit-column CSV/TSV manifest instead.

```bash
python prediction_pipeline/predict_pairs.py \
  --pairs prediction_pipeline/examples/pair_names.txt \
  --ahd-reference prediction_pipeline/examples/ahds.fasta \
  --rgg-reference prediction_pipeline/examples/rggs.fasta \
  --run-name example_pair_text
```

**B. Pair names with reference sequences** (`examples/pair_names.tsv`):

```bash
python prediction_pipeline/predict_pairs.py \
  --pairs prediction_pipeline/examples/pair_names.tsv \
  --ahd-reference prediction_pipeline/examples/ahds.fasta \
  --rgg-reference prediction_pipeline/examples/rggs.fasta \
  --run-name example_names
```

A missing inline sequence is resolved exactly from the corresponding registry. An inline sequence that conflicts with its named registry record is rejected.

**C. Full FASTA cross product**:

```bash
python prediction_pipeline/predict_pairs.py \
  --ahd-fasta prediction_pipeline/examples/ahds.fasta \
  --rgg-fasta prediction_pipeline/examples/rggs.fasta \
  --cross-product --run-name example_fasta_product
```

**D. Two name lists resolved against references**:

```bash
python prediction_pipeline/predict_pairs.py \
  --ahd-list prediction_pipeline/examples/ahd_names.txt \
  --rgg-list prediction_pipeline/examples/rgg_names.txt \
  --ahd-reference prediction_pipeline/examples/ahds.fasta \
  --rgg-reference prediction_pipeline/examples/rggs.fasta \
  --cross-product --run-name example_list_product
```

Reference arguments can be repeated. A reference can also be CSV/TSV with `name,sequence` columns. FASTA identifiers are the first whitespace-delimited header token, as in the historical reader; descriptions are not identifiers. Names are case-sensitive and never fuzzy-matched.

Generated pair IDs follow the existing **RGG_name__AHD_name** convention. Names containing `__` are percent-escaped before constructing the ID to prevent collisions; original names are retained in separate columns. Explicit IDs bypass generated-ID construction.

To validate sequences/features without loading any model or importing TensorFlow, add `--features-only`. This writes prepared inputs and audits, not prediction files. No simulation targets or RDFs are needed.

## Prepare once, predict later without repeating preparation

In your feature-preparation environment, run:

```bash
python prediction_pipeline/predict_pairs.py \
  --pairs inputs/pair_names.txt \
  --ahd-reference inputs/all_sequences.fasta \
  --rgg-reference inputs/all_sequences.fasta \
  --features-only --run-name experiment_01_features
```

Using the same FASTA for both roles is supported when it contains all sequences with unique identifiers.

Then activate your existing TensorFlow model environment and reuse the completed run directory:

```bash
python prediction_pipeline/predict_pairs.py \
  --prepared-run prediction_pipeline/prediction_runs/experiment_01_features \
  --run-name experiment_01_predictions
```

This **does not resolve your input sequences again or recalculate their features**. It reads `prepared_pairs.csv`, checks it against the saved feature matrix, row mapping, sequence hashes and preparation provenance, then runs the frozen predictor. You do not need to supply the original pair-name file or FASTAs again. The small mandatory known-reference identity check still runs, including its 20 reference feature rows; this is separate from preparing your input pairs.

Keep the **whole preparation run directory**, including `run_manifest.json`, identity checks and QC CSVs, not just `prepared_pairs.csv`. `--prepared-run` takes that directory, not a single CSV, and cannot be combined with `--pairs`, FASTA options, `--features-only` or `--identity-only`. Prediction outputs are saved under the new run name, leaving the preparation run unchanged.

Existing completed preparation runs from the previous pipeline version are supported when the feature implementation, resources and saved table alignment match. Those older runs did not record output checksums, so their reuse is explicitly marked `legacy_run_cross_checks_no_original_output_hashes`. New runs record `prepared_output_hashes` and verify them before reuse. No tolerance or model parameter is changed when resuming.

## Frozen computation

Canonical function bodies come from:

- `fig5_final_model_validation_local/scripts/historical_patch_get_features.py`: `analyse_features`, `patch_pair_summaries`, `calc_SHD_fast`, `calc_SCD`, and their helpers.
- `fig5_final_model_validation_local/scripts/energy_funcs.py`: historical AH integration/potential helpers.
- `s8_extended_controls/scripts/run_new_s8_control_predictions.py`: unchanged `predict`, `soft_gate`, `sha` functions, now under `vendor/accepted_predictor.py`.

No runtime import depends on those historical scripts. Vendored function bodies are retained verbatim; CLI setup was replaced by a library initializer. Optional Numba acceleration can be absent: the identical Python function bodies then run without JIT. Source and packaged hashes are in `resources/SOURCE_PROVENANCE.json` and are verified on each run.

Patch1 is RG/RGG; patch2 is AHD. Patch width 5, step 1, Cartesian patch-pair product, population SD (`ddof=0`). AH integrates the historical scaled potential from `2**(1/6)*sigma_pair` to 2.0 nm using epsilon `0.2*4.184`. These are sequence-derived descriptors. Global charge uses the stored residue charge convention; patch SCD uses the historical R/K = +1, D/E = -1, others = 0 convention, with no terminal charges. Histidine is neutral under this frozen feature convention. No pH-dependent charge correction is invented.

The stored base scaler, five base MLPs, residual scaler/network and manifest calibration are loaded. The accepted formula is:

```text
base = mean(five frozen MLP member predictions)
gate = sigmoid((minus_NCP - stored_gate_center) / stored_gate_scale)
weighted_residual = stored_lambda * gate * raw_residual
applied_correction = stored_lambda * gate * clip(raw_residual, -stored_cap, stored_cap)
final_hybrid_prediction = base + applied_correction
```

The cap is applied to the **raw residual before weighting/gating**. Values come from the accepted manifest; they are not fitted or inferred from new data. Prediction is `log10(-B22)` and is never back-transformed.

## Output schema

Every run is saved under `prediction_pipeline/prediction_runs/<run_name>/`.

`predictions.csv` contains the following ordered blocks:

1. `pair_id, AHD_name, RGG_name, AHD_sequence, RGG_sequence, AHD_length, RGG_length, AHD_sequence_sha256, RGG_sequence_sha256`
2. The 13 public descriptors, in this exact model order:
   `minus_NCP, mean_lambda_RGG, mean_lambda_AHD, SHD_RGG, SHD_AHD, local_charge_complementarity_std, scd_RGG_patch_std, scd_AHD_patch_std, rg_fraction_RGG_patch_std, aromatic_fraction_AHD_patch_std, shd_RGG_patch_std, shd_AHD_patch_std, ah_energy_patch_std`
3. `base_model_1, base_model_2, base_model_3, base_model_4, base_model_5, base_prediction, raw_residual_prediction, residual_gate, weighted_residual, applied_correction, final_hybrid_prediction, lambda_final, residual_cap`
4. `input_row, failure_reason, duplicate_pair_id, duplicate_sequence_pair, missing_sequence, invalid_residue, short_for_patch_features, status, AHD_duplicate_sequence_under_other_name, RGG_duplicate_sequence_under_other_name, high_NCP, n_features_outside_development_range, features_outside_development_range, n_features_outside_development_robust_range, features_outside_development_robust_range, feature_warnings`

`predictions_minimal.csv` contains exactly:

```text
pair_id,AHD_name,RGG_name,AHD_sequence,RGG_sequence,minus_NCP,base_prediction,applied_correction,final_hybrid_prediction
```

Other files:

| File | Contents |
|---|---|
| `feature_matrix.csv` | Exactly 13 public descriptor columns, in authoritative model order; no ID or index column |
| `feature_row_ids.csv` | Row-to-pair mapping; `feature_row` is zero-based |
| `prepared_pairs.csv` | Valid sequences, features and applicability flags |
| `resolved_pairs.csv` | All requested pairs, including failures |
| `rejected_pairs.csv` | All rejected pairs and reasons; header retained when empty |
| `sequence_registry_audit.csv` | Registry name/sequence duplicates and sequence validation |
| `pipeline_identity_test.csv/.txt` | Per-feature/prediction maximum deviations, tolerances and pass/fail |
| `run_manifest.json` | Input/model/code/resource hashes, calibration, counts, versions, command, timestamp, Git commit and status |
| `qc_summary.txt` | Human-readable run status and counts |
| `error_traceback.txt` | Created only when a run fails |

The historical/internal compatibility names and their public aliases are documented in `config/final_feature_schema.json`. The public descriptor `minus_NCP` is the negative net-charge product, `-Q_RGG*Q_AHD`; `local_charge_complementarity_std` and `ah_energy_patch_std` denote local charge-complementarity heterogeneity and local Ashbaugh–Hatch interaction-energy heterogeneity.

## Validation and numerical tolerances

Identity uses 20 unchanged accepted development references bundled with exact sequences and features. New data cannot change references or calibration. Comparisons use `rtol=0`:

| Quantity | Absolute tolerance |
|---|---:|
| All 13 descriptors | 1e-10 |
| Base prediction and gate | 1e-10 |
| Raw TensorFlow residual | 1e-7 |
| Applied correction and final prediction | 1e-8 |

The float32 backend allowance is based on the existing accepted S8 server identity audit (maximum raw discrepancy 4.8936e-8; final 1.6199e-9), not new-data performance. Every actual deviation is reported. Larger discrepancies stop inference; no automatic tolerance relaxation or retraining occurs.

Sequence whitespace is removed and residues uppercased. Empty sequences, nonstandard residues, lengths below five, missing/ambiguous names and duplicate pair IDs are rejected explicitly. Identical sequence pairs under different unique IDs remain and are flagged. Biological observed mean-lambda ranges and development min/max or 0.5th/99.5th-percentile warnings are descriptive only; out-of-range rows are not discarded.

Local preparation tests, without inference:

```bash
python -m unittest discover -s prediction_pipeline/tests -v
```

Server inference test, explicitly enabled:

```bash
RUN_FROZEN_MODEL_IDENTITY=1 python -m unittest discover -s prediction_pipeline/tests -v
```

Use the existing server TensorFlow/scikit-learn environment. `requirements-features.txt` and `requirements-inference.txt` list dependencies; they are not an instruction to upgrade an established environment. The local source manifest reports older sklearn/TensorFlow versions and differs from the accepted server manifest. The server's actual versions are recorded per run; serialized-object compatibility is confirmed by the mandatory identity check.
