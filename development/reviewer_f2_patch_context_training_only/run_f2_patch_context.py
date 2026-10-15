#!/usr/bin/env python
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reviewer_f2_patch_context_training_only"; OUT.mkdir(exist_ok=True)
INFILE=ROOT/"sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
PATCH=ROOT/"patches/datasets/features_sim_pairs_for_training.csv"
FOLD=ROOT/"reviewer_reduced_model_ablation_training_only/cv_fold_assignments.csv"
TARGET="B22_corr1_log10"; REPEATS=[42,43,44,45,46]; ENSEMBLE=[0,1,2,3,4]
F2=["Opposite_Charge_Number","mean_lambda_RGG","mean_lambda_TAD","SHD_RGG","SHD_TAD"]
MODELS={"F2":F2,"F2-P1":F2+["opp_charge_std"],"F2-P7":F2+["opp_charge_std","scd_patch1_std","scd_patch2_std","rg_fraction_patch1_std","aromatic_fraction_patch2_std","shd_patch1_std","shd_patch2_std"],"F2-P8":F2+["opp_charge_std","scd_patch1_std","scd_patch2_std","rg_fraction_patch1_std","aromatic_fraction_patch2_std","shd_patch1_std","shd_patch2_std","ah_energy_std"]}

def met(y,p): return {"R2":r2_score(y,p),"RMSE":mean_squared_error(y,p)**.5,"MAE":mean_absolute_error(y,p),"Pearson":stats.pearsonr(y,p).statistic,"Spearman":stats.spearmanr(y,p).statistic}
def fit(a,y,b):
 s=StandardScaler(); a=s.fit_transform(a); b=s.transform(b); ps=[]
 for seed in ENSEMBLE:
  m=MLPRegressor(activation="tanh",alpha=.01,hidden_layer_sizes=(128,64,32,16),learning_rate_init=.01,max_iter=500,solver="adam",random_state=seed); m.fit(a,y); ps.append(m.predict(b))
 return np.mean(ps,axis=0)
def main():
 base=pd.read_csv(INFILE); patch=pd.read_csv(PATCH)
 assert base['name'].is_unique
 assert patch['name'].is_unique
 assert len(base)==917 and len(patch)==917
 bset=set(base['name']); pset=set(patch['name'])
 print(f'base IDs = {len(bset)}; patch IDs = {len(pset)}; intersect = {len(bset & pset)}; base-only = {len(bset-pset)}; patch-only = {len(pset-bset)}')
 assert bset == pset
 df=base.copy(); df['row_id']=np.arange(len(df))
 df["mean_lambda_RGG"]=df["lambda_RGG"]/df["RGG_length"]; df["mean_lambda_TAD"]=df["lambda_TAD"]/df["TAD_length"]
 patch_only=[c for c in sorted(set(MODELS['F2-P8'])-set(F2)) if c in patch.columns]
 patch_aligned=patch[['name']+patch_only].copy()
 df=df.merge(patch_aligned,on='name',how='left',validate='one_to_one',sort=False)
 assert len(df)==917 and df[patch_only].notna().all().all()
 req=[TARGET]+sorted(set(sum(MODELS.values(),[]))); df=df[np.isfinite(df[req].to_numpy(float)).all(axis=1)].reset_index(drop=True); y=df[TARGET].to_numpy(float)
 folds=pd.read_csv(FOLD); folds=folds[folds.row_id.isin(df.row_id)]
 rows=[]; preds=[]
 for seed in REPEATS:
  fs=folds[folds.repeat_seed==seed]
  for name,cols in MODELS.items():
   p=np.empty(len(df))
   for fold in sorted(fs.fold.unique()):
    vi=df.index[df.row_id.isin(fs.loc[fs.fold==fold,'row_id'])].to_numpy(); ti=df.index[~df.row_id.isin(fs.loc[fs.fold==fold,'row_id'])].to_numpy(); p[vi]=fit(df.iloc[ti][cols].to_numpy(float),y[ti],df.iloc[vi][cols].to_numpy(float))
   rows.append({"repeat_seed":seed,"model":name,**met(y,p)}); preds.append(pd.DataFrame({"row_id":df.row_id,"repeat_seed":seed,"model":name,"prediction":p,"true":y}))
 pd.DataFrame(rows).to_csv(OUT/"cv_metrics_by_repeat.csv",index=False); pd.concat(preds).to_csv(OUT/"cv_fold_predictions.csv",index=False)
 perf=pd.DataFrame(rows).groupby('model').agg(['mean','std']).reset_index(); perf.to_csv(OUT/"cv_metrics_summary.csv",index=False)
 comps=[]
 for hi,lo in [("F2-P1","F2"),("F2-P7","F2"),("F2-P8","F2"),("F2-P7","F2-P1"),("F2-P8","F2-P7")]:
  for seed in REPEATS:
   a=next(r for r in rows if r['model']==hi and r['repeat_seed']==seed); b=next(r for r in rows if r['model']==lo and r['repeat_seed']==seed); comps.append({'comparison':f'{hi}_minus_{lo}','repeat_seed':seed,**{f'delta_{k}':a[k]-b[k] for k in ['R2','RMSE','MAE','Pearson','Spearman']}})
 pd.DataFrame(comps).to_csv(OUT/"paired_comparisons_by_repeat.csv",index=False)
 (OUT/"input_provenance.txt").write_text(f"917-pair training-only F2 patch extension. Input global: {INFILE}\nInput frozen patch table: {PATCH}\nNo held-out matrix accessed. Historical orientation: patch1=RGG, patch2=TAD.\nExact existing pipeline repeats={REPEATS}, ensemble seeds={ENSEMBLE}, shared fold assignments={FOLD}.\n")
 (OUT/"model_definitions.csv").write_text("model,features\n"+"\n".join(f"{k},{';'.join(v)}" for k,v in MODELS.items())+"\n")
 print(pd.DataFrame(rows).groupby('model')[['R2','RMSE','MAE','Pearson','Spearman']].mean())
if __name__=='__main__': main()
