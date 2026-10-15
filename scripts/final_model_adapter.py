#!/usr/bin/env python3
"""Snakemake bridge to the identity-checked final frozen predictor."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import pandas as pd
from configuration import PUBLIC_FEATURES

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['features','predict']);p.add_argument('--pairs');p.add_argument('--fasta');p.add_argument('--features');p.add_argument('--output',required=True);p.add_argument('--ranked');a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if a.mode=='features':
        from sequence_io import registry_records
        from feature_engineering import calculate_features
        from qc import identity_test
        identity_test(out.parent)
        registry={}
        for name,seq in registry_records(Path(a.fasta)):
            if name in registry and registry[name]!=seq: raise ValueError(f'Conflicting sequence: {name}')
            registry[name]=seq
        pairs=pd.read_csv(a.pairs)
        if not pairs.pair_id.is_unique: raise ValueError('Duplicate pair IDs')
        data=pd.DataFrame({'pair_id':pairs.pair_id,'RGG_name':pairs.idr_a,'AHD_name':pairs.idr_b,'RGG_sequence':pairs.idr_a.map(registry),'AHD_sequence':pairs.idr_b.map(registry)})
        for col in ['RGG_sequence','AHD_sequence']:
            if data[col].isna().any() or not data[col].str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{5,}').all(): raise ValueError('Missing/invalid sequence or sequence shorter than five residues')
        pd.concat([data,calculate_features(data)],axis=1).to_csv(out,index=False)
    else:
        from model_loader import FrozenPredictor
        from qc import identity_test
        model=FrozenPredictor();identity_test(out.parent,model)
        data=pd.read_csv(a.features)
        if not data.pair_id.is_unique: raise ValueError('Duplicate pair IDs')
        result=pd.concat([data,model.predict(data[PUBLIC_FEATURES],data.pair_id)],axis=1)
        result.to_csv(out,index=False)
        ranked=Path(a.ranked);ranked.parent.mkdir(parents=True,exist_ok=True)
        result.sort_values('final_hybrid_prediction',ascending=False).to_csv(ranked,index=False)
if __name__=='__main__':main()
