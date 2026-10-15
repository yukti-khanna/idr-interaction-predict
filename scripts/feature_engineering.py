"""Adapter to the literal historical feature bodies; no alternative formulas."""
import importlib.util
import numpy as np
import pandas as pd
from configuration import PIPELINE_ROOT, INTERNAL_FEATURES, INTERNAL_TO_PUBLIC, PUBLIC_FEATURES, verify_resources
from vendor import historical_features as historical

_INITIALIZED = False
_GLOBAL = None

def calculate_features(pairs):
    global _INITIALIZED, _GLOBAL
    verify_resources()
    if not _INITIALIZED:
        historical.initialize(PIPELINE_ROOT / 'resources/residues.csv')
        # Same canonical functions in an isolated module namespace with the historical global table.
        spec = importlib.util.spec_from_file_location('vendor.global_features', PIPELINE_ROOT/'scripts/vendor/historical_features.py')
        _GLOBAL = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_GLOBAL)
        _GLOBAL.initialize(PIPELINE_ROOT/'resources/global_residues.csv')
        _INITIALIZED = True
    if len(pairs)==0: return pd.DataFrame(columns=PUBLIC_FEATURES, index=pairs.index)
    # Private identifiers avoid parsing user-supplied names; biological role stays fixed.
    names=[]; sequences={}
    for i, row in enumerate(pairs.to_dict('records')):
        rgg=f'RGG{i:09d}'; ahd=f'AHD{i:09d}'
        sequences[rgg]=row['RGG_sequence']; sequences[ahd]=row['AHD_sequence']
        names.append(f'{rgg}__{ahd}')
    raw=pd.DataFrame(historical.analyse_features(names,historical.residue_properties,sequences,patch_size=5,shard=0,nshards=1))
    if len(raw)!=len(pairs) or not raw.name.is_unique or set(raw.name)!=set(names):
        raise RuntimeError('Historical feature generator returned incomplete or duplicate pairs')
    raw=raw.set_index('name').loc[names].reset_index()
    # The frozen training input merged authoritative globals with separate historical patches.
    # Recover the four global hydropathy columns with their original residue table.
    for role, internal_role in [('RGG','RGG'),('AHD','TAD')]:
        indices=[_GLOBAL.encode_seq(seq) for seq in pairs[role+'_sequence']]
        raw['mean_lambda_'+internal_role]=[float(_GLOBAL.per_res_props(ix)[0].sum())/len(ix) for ix in indices]
        raw['SHD_'+internal_role]=[_GLOBAL.calc_SHD_fast(ix) for ix in indices]
    expected=-raw['charge_RGG'].to_numpy()*raw['charge_TAD'].to_numpy()
    if not np.allclose(raw['Opposite_Charge_Number'],expected,rtol=0,atol=1e-10):
        raise RuntimeError('Negative net-charge product identity failed')
    features=raw[INTERNAL_FEATURES].rename(columns=INTERNAL_TO_PUBLIC)
    features.index=pairs.index
    return features
