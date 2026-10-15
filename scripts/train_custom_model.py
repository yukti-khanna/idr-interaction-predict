#!/usr/bin/env python3
"""Train a separately named model on user-supplied sequences and fixed splits."""
from pathlib import Path
import argparse,json,platform,sys,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from scipy.special import expit
from configuration import PUBLIC_FEATURES,PIPELINE_ROOT,verify_resources,sha256
from qc import identity_test
from custom_model import read_sequences,metrics,predict_components
from feature_engineering import calculate_features

PARAMS=dict(activation='tanh',alpha=.01,hidden_layer_sizes=(128,64,32,16),learning_rate_init=.01,solver='adam',max_iter=500)
SEEDS=[0,1,2,3,4]
def ensemble(X,y):
    scaler=StandardScaler().fit(X);z=scaler.transform(X)
    models=[MLPRegressor(**PARAMS,random_state=seed).fit(z,y) for seed in SEEDS]
    return {'scaler':scaler,'models':models}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--target-column',required=True);p.add_argument('--target-description',required=True,help='Units, correction and transformation; target values are used exactly as supplied');p.add_argument('--mode',choices=['base','hybrid'],default='base');p.add_argument('--split-unit',choices=['sequence','pair'],default='sequence');p.add_argument('--residual-epochs',type=int,default=400);a=p.parse_args()
    out=a.output.resolve();protected=(PIPELINE_ROOT/'models/final_frozen').resolve()
    if out==protected or protected in out.parents or out in protected.parents:raise ValueError('Custom training cannot write to or contain the frozen-model directory')
    if out.exists():raise ValueError('Output must be a new directory; existing models are never overwritten')
    if a.residual_epochs<1:raise ValueError('epochs must be positive')
    if not a.target_description.strip() or 'REPLACE_WITH' in a.target_description:raise ValueError('Provide the actual target units, correction and transformation')
    df,overlap=read_sequences(a.data,True,a.split_unit)
    if a.target_column not in df:raise ValueError('Target column missing')
    tr=df.split.eq('train');va=df.split.eq('validation');te=df.split.eq('test')
    if tr.sum()<10 or va.sum()<3:raise ValueError('Need >=10 training rows and >=3 validation rows')
    # Only training/validation targets are consumed during model construction.
    yt=pd.to_numeric(df.loc[tr,a.target_column],errors='raise').to_numpy(float);yv=pd.to_numeric(df.loc[va,a.target_column],errors='raise').to_numpy(float)
    if not np.isfinite(yt).all() or not np.isfinite(yv).all():raise ValueError('Non-finite training/validation targets')
    if a.mode=='hybrid':
        # Verify dependency availability before creating an incomplete run.
        import tensorflow as tf
    out.mkdir(parents=True);identity_test(out)
    features=calculate_features(df);X=features.to_numpy(float)
    if not np.isfinite(X).all():raise ValueError('Non-finite features')
    bundle=ensemble(X[tr],yt)
    manifest={'model_kind':'user_trained','mode':a.mode,'feature_order':PUBLIC_FEATURES,'target_column':a.target_column,'target_description':a.target_description,'input_sha256':sha256(a.data),'split_unit':a.split_unit,'split_counts':df.split.value_counts().to_dict(),'sequence_overlap':overlap,'MLP_parameters':PARAMS,'ensemble_seeds':SEEDS,'feature_adapter_sha256':sha256(PIPELINE_ROOT/'scripts/feature_engineering.py'),'resource_hashes':verify_resources()['bundled_files'],'gate_center':40.,'gate_scale':12.,'lambda_final':0.,'residual_cap':0.,'residual_epochs':a.residual_epochs,'calibration':('validation-only bounded least squares lambda; training OOF residual absolute 85th percentile cap' if a.mode=='hybrid' else 'none; base ensemble only'),'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib']},'python':platform.python_version(),'command':sys.argv}
    residual=None
    if a.mode=='hybrid':
        # Residual targets must come from base predictions out of fold within TRAIN.
        xt=X[tr];oof=np.empty(len(yt));fold_id=np.empty(len(yt),int)
        for fold,(fit_idx,val_idx) in enumerate(KFold(5,shuffle=True,random_state=42).split(xt)):
            b=ensemble(xt[fit_idx],yt[fit_idx]);oof[val_idx]=np.mean([m.predict(b['scaler'].transform(xt[val_idx])) for m in b['models']],axis=0);fold_id[val_idx]=fold
        high=features.loc[tr,'minus_NCP'].to_numpy(float)>=40
        if high.sum()<10:raise ValueError('Hybrid requires >=10 training pairs with minus_NCP >=40; use base mode for this dataset')
        import tensorflow as tf
        tf.keras.utils.set_random_seed(20260916)
        scaler=StandardScaler().fit(xt[high]);bundle['residual_scaler']=scaler
        residual=tf.keras.Sequential([tf.keras.layers.Input(shape=(13,)),tf.keras.layers.Dense(16,activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(5e-4)),tf.keras.layers.Dense(8,activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(5e-4)),tf.keras.layers.Dense(1)])
        residual.compile(optimizer=tf.keras.optimizers.Adam(2e-4),loss=tf.keras.losses.Huber(delta=.75))
        residual.fit(scaler.transform(xt[high]),(yt-oof)[high],sample_weight=expit((features.loc[tr,'minus_NCP'].to_numpy(float)[high]-40)/12),epochs=a.residual_epochs,batch_size=32,verbose=0)
        cap=float(np.percentile(np.abs((yt-oof)[high]),85));manifest['residual_cap']=cap
        raw=residual.predict(scaler.transform(X[va]),verbose=0).reshape(-1)
        c=expit((features.loc[va,'minus_NCP'].to_numpy(float)-40)/12)*np.clip(raw,-cap,cap)
        base=predict_components(bundle,features.loc[va],manifest).base_prediction.to_numpy()
        manifest['lambda_final']=float(np.clip(np.dot(c,yv-base)/np.dot(c,c),0,1)) if np.dot(c,c)>0 else 0.
        manifest['software']['tensorflow']=tf.__version__
        pd.DataFrame({'pair_id':df.loc[tr,'pair_id'],'fold':fold_id,'target':yt,'base_OOF_prediction':oof}).to_csv(out/'training_base_oof.csv',index=False)
        residual.save(out/'residual.keras')
    joblib.dump(bundle,out/'ensemble.joblib');(out/'MODEL_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    prediction=predict_components(bundle,features,manifest,residual)
    rows=[]
    for split in ['train','validation','test']:
        mask=df.split.eq(split)
        if not mask.any():continue
        # Test targets are evaluated only after the model and calibration are saved.
        y=pd.to_numeric(df.loc[mask,a.target_column],errors='raise').to_numpy(float)
        if not np.isfinite(y).all():raise ValueError(f'Non-finite {split} targets')
        rows.append({'split':split,'interpretation':'in_sample' if split=='train' else 'calibration_set' if split=='validation' and a.mode=='hybrid' else 'evaluation',**metrics(y,prediction.loc[mask,'final_prediction'].to_numpy())})
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.concat([df,features,prediction],axis=1).to_csv(out/'evaluation_predictions.csv',index=False)
    files=['MODEL_MANIFEST.json','ensemble.joblib']+(['residual.keras'] if residual is not None else [])
    (out/'SHA256SUMS.json').write_text(json.dumps({n:sha256(out/n) for n in files},indent=2))
    print(f'Custom model saved in {out}; supplied frozen model untouched.')
if __name__=='__main__':main()
