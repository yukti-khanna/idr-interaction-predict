#!/usr/bin/env python3
"""Predict with the frozen Figure 5 final hybrid model."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def soft_gate(x, center, scale):
    return 1.0/(1.0+np.exp(-((np.asarray(x,dtype=float)-center)/scale)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--model-dir', default=str(Path(__file__).resolve().parent))
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    model_dir=Path(args.model_dir)
    manifest=json.loads((model_dir/'MODEL_MANIFEST.json').read_text())
    df=pd.read_csv(args.features)
    required=['name']+manifest['feature_order']
    missing=[c for c in required if c not in df.columns]
    if missing: raise SystemExit(f'Missing columns: {missing}')
    base=joblib.load(model_dir/'base_model_ensemble.joblib')
    X=df[manifest['feature_order']].to_numpy(float)
    Xb=base['scaler'].transform(X)
    base_pred=np.mean([m.predict(Xb) for m in base['models']], axis=0)
    prep=joblib.load(model_dir/'residual_preprocessor.joblib')
    import tensorflow as tf
    residual=tf.keras.models.load_model(model_dir/'residual_nn.keras')
    Xr=prep['scaler'].transform(X)
    raw=residual.predict(Xr, verbose=0).reshape(-1)
    minus_ncp=df[manifest['minus_NCP_feature']].to_numpy(float)
    gate=soft_gate(minus_ncp, manifest['gate_center'], manifest['gate_scale'])
    clipped=np.clip(raw, -manifest['residual_cap'], manifest['residual_cap'])
    uncapped=manifest['lambda_final']*gate*raw
    applied=manifest['lambda_final']*gate*clipped
    final=base_pred+applied
    out=pd.DataFrame({'name':df['name'], 'base_prediction':base_pred, 'raw_residual_prediction':raw, 'gate_value':gate, 'lambda_final':manifest['lambda_final'], 'uncapped_correction':uncapped, 'applied_correction':applied, 'final_hybrid_prediction':final})
    if not np.allclose(out['final_hybrid_prediction'], out['base_prediction']+out['applied_correction'], atol=1e-10):
        raise SystemExit('Prediction identity check failed')
    out.to_csv(args.out, index=False)
if __name__=='__main__': main()
