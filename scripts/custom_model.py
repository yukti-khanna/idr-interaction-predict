"""Independent user-trained models; never changes the manuscript frozen model."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr,spearmanr
from configuration import PUBLIC_FEATURES,PIPELINE_ROOT,verify_resources,sha256
from feature_engineering import calculate_features

def read_sequences(path,training=False,split_unit='sequence'):
    df=pd.read_csv(path,keep_default_na=False)
    if 'rAHD_sequence' in df:df=df.rename(columns={'rAHD_sequence':'AHD_sequence'})
    required=['pair_id','AHD_sequence','RGG_sequence']+(['split'] if training else [])
    if any(c not in df for c in required):raise ValueError(f'Required columns: {required}; use rAHD_sequence for the public input')
    if df.empty or not df.pair_id.astype(str).str.strip().ne('').all() or not df.pair_id.is_unique:raise ValueError('Pair IDs must be nonempty and unique')
    for c in ['AHD_sequence','RGG_sequence']:
        df[c]=df[c].astype(str).str.replace(r'\s+','',regex=True).str.upper()
        if not df[c].str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{5,}').all():raise ValueError('Sequences must contain standard amino acids and have length >=5')
    if df.duplicated(['AHD_sequence','RGG_sequence']).any():raise ValueError('Duplicate sequence pairs; consolidate replicates before training/prediction')
    overlap=[]
    if training:
        if not set(df.split)<={'train','validation','test'} or not {'train','validation'}<=set(df.split):raise ValueError('split must contain train and validation; test is optional')
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            sa=set(df.loc[df.split.eq(a),['AHD_sequence','RGG_sequence']].values.ravel());sb=set(df.loc[df.split.eq(b),['AHD_sequence','RGG_sequence']].values.ravel())
            overlap.append({'split_a':a,'split_b':b,'shared_sequences':len(sa&sb)})
        if split_unit=='sequence' and any(x['shared_sequences'] for x in overlap):raise ValueError('Sequences overlap across splits; supply sequence-exclusive splits, or explicitly use --split-unit pair and report this limitation')
    return df,overlap

def metrics(y,p):
    return {'n':len(y),'R2':float(r2_score(y,p)) if len(y)>1 else None,'RMSE':float(mean_squared_error(y,p)**.5),'MAE':float(mean_absolute_error(y,p)),'Pearson':float(pearsonr(y,p)[0]) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else None,'Spearman':float(spearmanr(y,p).statistic) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else None,'mean_prediction_minus_target':float(np.mean(p-y))}

def predict_components(bundle,features,manifest,residual=None):
    X=features[PUBLIC_FEATURES].to_numpy(float);base=np.mean([m.predict(bundle['scaler'].transform(X)) for m in bundle['models']],axis=0)
    raw=np.zeros(len(X));gate=np.zeros(len(X));correction=np.zeros(len(X))
    if residual is not None:
        raw=residual.predict(bundle['residual_scaler'].transform(X),verbose=0).reshape(-1)
        gate=expit((features.minus_NCP.to_numpy(float)-manifest['gate_center'])/manifest['gate_scale'])
        correction=manifest['lambda_final']*gate*np.clip(raw,-manifest['residual_cap'],manifest['residual_cap'])
    return pd.DataFrame({'base_prediction':base,'raw_residual_prediction':raw,'gate_value':gate,'applied_correction':correction,'final_prediction':base+correction},index=features.index)

def load_custom(directory):
    import joblib
    directory=Path(directory);manifest=json.loads((directory/'MODEL_MANIFEST.json').read_text())
    hashes=json.loads((directory/'SHA256SUMS.json').read_text())
    required={'MODEL_MANIFEST.json','ensemble.joblib'} | ({'residual.keras'} if manifest['mode']=='hybrid' else set())
    if set(hashes)!=required:raise ValueError('Incomplete or unexpected custom artifact manifest')
    for n,h in hashes.items():
        if sha256(directory/n)!=h:raise ValueError(f'Custom artifact checksum mismatch: {n}')
    if manifest['feature_adapter_sha256']!=sha256(PIPELINE_ROOT/'scripts/feature_engineering.py'):raise ValueError('Custom feature adapter differs from training')
    if manifest['feature_order']!=PUBLIC_FEATURES or manifest['resource_hashes']!=verify_resources()['bundled_files']:raise ValueError('Custom model feature implementation/resources do not match')
    bundle=joblib.load(directory/'ensemble.joblib');residual=None
    if manifest['mode']=='hybrid':
        import tensorflow as tf
        residual=tf.keras.models.load_model(directory/'residual.keras',compile=False)
    return bundle,manifest,residual
