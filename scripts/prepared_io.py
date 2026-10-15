"""Validate and reuse a completed preparation run without recalculating features."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from configuration import PIPELINE_ROOT, PUBLIC_FEATURES, SCHEMA, sha256, verify_resources

PREPARED_FILES = ['prepared_pairs.csv', 'feature_matrix.csv', 'feature_row_ids.csv',
                  'resolved_pairs.csv', 'rejected_pairs.csv', 'sequence_registry_audit.csv']

def load_prepared_run(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/'run_manifest.json').read_text())
    if manifest.get('status') not in ['features_prepared_prediction_pending', 'completed', 'completed_with_rejections']:
        raise ValueError('Prepared run did not finish successfully')
    if manifest.get('feature_order') != PUBLIC_FEATURES:
        raise ValueError('Prepared feature order does not match the final schema')
    if manifest.get('expected_model_manifest_sha256') != SCHEMA['expected_manifest_sha256']:
        raise ValueError('Prepared run targets a different model manifest')
    current = verify_resources()['bundled_files']
    previous = manifest.get('canonical_resources', {}).get('bundled_files', {})
    if current != previous:
        raise ValueError('Canonical feature resources differ from the preparation run')
    recorded_code = manifest.get('pipeline_code_hashes', {})
    if recorded_code.get('feature_engineering.py') != sha256(PIPELINE_ROOT/'scripts/feature_engineering.py'):
        raise ValueError('Feature adapter differs from the preparation run; investigate before reuse')
    files = [directory/name for name in PREPARED_FILES]
    if not all(p.is_file() for p in files):
        raise ValueError('Prepared run is incomplete; keep the whole preparation run directory')
    hashes = manifest.get('prepared_output_hashes', {})
    if hashes:
        for name in PREPARED_FILES:
            if hashes.get(name) != sha256(directory/name):
                raise ValueError(f'Prepared output checksum mismatch: {name}')
    identity = pd.read_csv(directory/'pipeline_identity_test.csv')
    feature_checks = identity[identity.quantity.isin(PUBLIC_FEATURES)]
    if (len(feature_checks) != 13 or not feature_checks.quantity.is_unique
            or not feature_checks.passed.eq(True).all()
            or not np.isfinite(feature_checks.max_absolute_difference).all()
            or not feature_checks.max_absolute_difference.le(1e-10).all()):
        raise ValueError('Preparation did not pass all 13 known-row feature checks')
    string_cols = {c:str for c in ['pair_id','AHD_name','RGG_name','AHD_sequence','RGG_sequence',
                                   'AHD_sequence_sha256','RGG_sequence_sha256','failure_reason']}
    ready = pd.read_csv(directory/'prepared_pairs.csv', dtype=string_cols, keep_default_na=False, float_precision='round_trip')
    matrix = pd.read_csv(directory/'feature_matrix.csv', float_precision='round_trip')
    ids = pd.read_csv(directory/'feature_row_ids.csv', dtype={'pair_id':str})
    requested = pd.read_csv(directory/'resolved_pairs.csv', dtype=string_cols, keep_default_na=False)
    rejected = pd.read_csv(directory/'rejected_pairs.csv', dtype=string_cols, keep_default_na=False)
    if ready.empty or not ready.pair_id.is_unique:
        raise ValueError('Prepared pair IDs must be nonempty and unique')
    if matrix.columns.tolist() != PUBLIC_FEATURES or len(matrix) != len(ready):
        raise ValueError('Prepared feature matrix schema/count mismatch')
    if (ids.pair_id.tolist() != ready.pair_id.tolist()
            or ids.feature_row.tolist() != list(range(len(ready)))
            or ids.input_row.tolist() != ready.input_row.tolist()):
        raise ValueError('Prepared pair-to-feature alignment mismatch')
    values = ready[PUBLIC_FEATURES].to_numpy(float)
    if not np.isfinite(values).all() or not np.array_equal(values, matrix.to_numpy(float)):
        raise ValueError('Prepared feature matrix disagrees with prepared_pairs.csv or has non-finite values')
    if (len(ready) != manifest.get('number_feature_rows')
            or len(requested) != manifest.get('number_requested_pairs')
            or len(rejected) != manifest.get('number_rejected')
            or len(requested) != len(ready)+len(rejected)):
        raise ValueError('Prepared run row counts do not reconcile')
    if set(ready.pair_id) & set(rejected.pair_id):
        raise ValueError('Prepared and rejected pair IDs overlap')
    for role in ['AHD','RGG']:
        if not ready[role+'_sequence'].str.len().eq(ready[role+'_length']).all():
            raise ValueError(f'{role} sequence length mismatch')
        sequence_hash = ready[role+'_sequence'].map(lambda s:hashlib.sha256(s.encode()).hexdigest())
        if not sequence_hash.eq(ready[role+'_sequence_sha256']).all():
            raise ValueError(f'{role} sequence checksum mismatch')
    # Reconcile metadata by original row identity; duplicate rejected pair IDs are allowed.
    if not requested.input_row.is_unique or not ready.input_row.is_unique or not rejected.input_row.is_unique:
        raise ValueError('Duplicate input-row identifiers in preparation audit')
    if set(ready.input_row) & set(rejected.input_row) or set(requested.input_row) != set(ready.input_row) | set(rejected.input_row):
        raise ValueError('Prepared/rejected rows do not partition the requested inputs')
    match = requested.set_index('input_row').loc[ready.input_row]
    for col in ['pair_id','AHD_name','RGG_name','AHD_sequence','RGG_sequence']:
        if match[col].tolist() != ready[col].tolist():
            raise ValueError(f'Prepared metadata differs from resolved input: {col}')
    audit = {'source_run_directory':str(directory), 'source_run_manifest_sha256':sha256(directory/'run_manifest.json'),
             'original_input_files':manifest.get('input_files',[]),
             'integrity_check':'stored_output_hashes_and_cross_checks' if hashes else 'legacy_run_cross_checks_no_original_output_hashes',
             'recalculated_input_features':False}
    return ready, requested, rejected, manifest, audit
