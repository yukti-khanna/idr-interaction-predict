"""Portable locations and the frozen public/internal feature mapping."""
import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
FIG5_ROOT = PIPELINE_ROOT
MODEL_DIR = FIG5_ROOT / 'models/final_frozen'
SCHEMA = json.loads((PIPELINE_ROOT / 'config/final_feature_schema.json').read_text())
PUBLIC_FEATURES = SCHEMA['public_order']
INTERNAL_FEATURES = SCHEMA['internal_order']
PUBLIC_TO_INTERNAL = dict(zip(PUBLIC_FEATURES, INTERNAL_FEATURES))
INTERNAL_TO_PUBLIC = dict(zip(INTERNAL_FEATURES, PUBLIC_FEATURES))

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def verify_resources():
    provenance = json.loads((PIPELINE_ROOT / 'resources/SOURCE_PROVENANCE.json').read_text())
    for relative, expected in provenance['bundled_files'].items():
        path = PIPELINE_ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Bundled canonical resource changed or missing: {relative}')
    return provenance
