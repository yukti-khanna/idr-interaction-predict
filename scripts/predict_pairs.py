#!/usr/bin/env python3
"""Sequence-based inference for the final rAHD–RG/RGG predictor."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
import numpy as np
import pandas as pd
from configuration import PIPELINE_ROOT, FIG5_ROOT, MODEL_DIR, SCHEMA, PUBLIC_FEATURES, sha256, verify_resources
from sequence_io import resolve_inputs, FLAGS
from feature_engineering import calculate_features
from qc import applicability, identity_test

META=['pair_id','AHD_name','RGG_name','AHD_sequence','RGG_sequence','AHD_length','RGG_length',
      'AHD_sequence_sha256','RGG_sequence_sha256']
MINIMAL=['pair_id','AHD_name','RGG_name','AHD_sequence','RGG_sequence','minus_NCP','base_prediction','applied_correction','final_hybrid_prediction']
PREDICTION_COLS=[f'base_model_{i}' for i in range(1,6)]+['base_prediction','raw_residual_prediction','residual_gate','weighted_residual','applied_correction','final_hybrid_prediction','lambda_final','residual_cap']

def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs',type=Path,help='CSV/TSV pair manifest, or headerless .txt with one RGG_name__AHD_name per line (sequences from reference FASTAs).')
    ap.add_argument('--prepared-run',type=Path,help='Reuse an existing completed preparation run directory; skip input resolution and input feature generation.')
    for role in ['ahd','rgg']:
        ap.add_argument(*(['--rahd-fasta','--ahd-fasta'] if role=='ahd' else ['--rgg-fasta']),dest=role+'_fasta',type=Path)
        ap.add_argument(*(['--rahd-list','--ahd-list'] if role=='ahd' else ['--rgg-list']),dest=role+'_list',type=Path)
        ap.add_argument(*(['--rahd-reference','--ahd-reference'] if role=='ahd' else ['--rgg-reference']),dest=role+'_reference',type=Path,action='append',default=[])
    ap.add_argument('--cross-product',action='store_true')
    ap.add_argument('--run-name',default=None)
    ap.add_argument('--features-only',action='store_true',help='Validate inputs and calculate features only; never imports TensorFlow or loads models.')
    ap.add_argument('--identity-only',action='store_true',help='Run known-row identity checks only, without new pair inputs.')
    return ap

def software_versions():
    result={'python':sys.version,'platform':platform.platform()}
    for name in ['numpy','pandas','scipy','numba','scikit-learn','joblib','tensorflow','keras']:
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]='not installed'
    return result

def validate_mode(ap,args):
    if args.prepared_run:
        if any([args.pairs,args.ahd_fasta,args.rgg_fasta,args.ahd_list,args.rgg_list,
                args.ahd_reference,args.rgg_reference,args.cross_product,args.features_only,args.identity_only]):
            ap.error('--prepared-run is prediction-only and cannot be combined with input or preparation options')
        return
    if args.identity_only:
        if args.pairs or args.ahd_fasta or args.rgg_fasta or args.ahd_list or args.rgg_list:
            ap.error('--identity-only cannot be combined with new pair inputs')
        return
    if args.pairs:
        if args.cross_product or args.ahd_list or args.rgg_list:
            ap.error('--pairs cannot be combined with --cross-product/name lists')
        return
    if not args.cross_product:ap.error('Supply --pairs, or --cross-product with two FASTAs or two name lists')
    if args.ahd_list or args.rgg_list:
        if not (args.ahd_list and args.rgg_list):ap.error('Both name lists are required')
        if args.ahd_fasta or args.rgg_fasta:ap.error('Use --*-reference with name lists')
    elif not (args.ahd_fasta and args.rgg_fasta):ap.error('Both FASTAs are required for the FASTA cross product')

def main():
    ap=parser(); args=ap.parse_args(); validate_mode(ap,args)
    name=args.run_name or datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name):ap.error('run-name must be a simple directory name without slashes')
    out=PIPELINE_ROOT/'results/runs'/name
    if out.exists():ap.error(f'Run already exists: {out}. Choose a new run name to preserve previous results.')
    out.mkdir(parents=True)
    manifest={'run_name':name,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'started',
              'mode':'prepared_inference' if args.prepared_run else 'features_only' if args.features_only else 'inference','feature_order':PUBLIC_FEATURES,
              'model_internal_feature_order':SCHEMA['internal_order'],'target':SCHEMA['target'],
              'model_directory_relative_to_repository':str(MODEL_DIR.relative_to(FIG5_ROOT)),
              'expected_model_manifest_sha256':SCHEMA['expected_manifest_sha256'],
              'software_versions':software_versions(),'input_files':[],'number_requested_pairs':0,'number_predicted':0,'number_rejected':0,
              'predictor_script_hash':sha256(PIPELINE_ROOT/'scripts/vendor/accepted_predictor.py'),
              'residue_property_table_sha256':{'patch':sha256(PIPELINE_ROOT/'resources/residues.csv'),'global':sha256(PIPELINE_ROOT/'resources/global_residues.csv')},
              'pipeline_code_hashes':{str(p.relative_to(PIPELINE_ROOT/'scripts')):sha256(p) for p in (PIPELINE_ROOT/'scripts').rglob('*.py')},
              'command':sys.argv}
    try:
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=FIG5_ROOT,capture_output=True,text=True)
        manifest['git_commit']=commit.stdout.strip() if commit.returncode==0 else None
        model_manifest=MODEL_DIR/'MODEL_MANIFEST.json'
        manifest['observed_model_manifest_sha256']=sha256(model_manifest) if model_manifest.is_file() else None
        manifest['canonical_resources']=verify_resources()
        requested=None; valid=None
        ready=None
        if args.prepared_run:
            from prepared_io import load_prepared_run, PREPARED_FILES
            ready,requested,rejected,previous,reuse_audit=load_prepared_run(args.prepared_run)
            manifest['prepared_input_provenance']=reuse_audit
            manifest['input_files']=[{'path':str((args.prepared_run/p).resolve()),'sha256':sha256(args.prepared_run/p)}
                                     for p in ['run_manifest.json','pipeline_identity_test.csv',*PREPARED_FILES]]
            for key in ['number_requested_pairs','number_rejected','number_AHD_sequences','number_RGG_sequences']:
                manifest[key]=previous[key]
            for filename in PREPARED_FILES:shutil.copyfile(args.prepared_run/filename,out/filename)
            valid=ready
            features=ready[PUBLIC_FEATURES].copy()
            print(f'Reusing prepared features for {len(ready)} pairs; no input preparation repeated.',flush=True)
        elif not args.identity_only:
            requested,registry,paths=resolve_inputs(args)
            manifest['input_files']=[{'path':str(Path(p).resolve()),'sha256':sha256(p)} for p in paths]
            manifest.update(number_requested_pairs=len(requested),number_rejected=int(requested.status.eq('rejected').sum()),
                            number_AHD_sequences=int(requested.AHD_name.nunique()),number_RGG_sequences=int(requested.RGG_name.nunique()))
            requested.to_csv(out/'resolved_pairs.csv',index=False)
            registry.to_csv(out/'sequence_registry_audit.csv',index=False)
            requested[requested.status.eq('rejected')].to_csv(out/'rejected_pairs.csv',index=False)
            valid=requested[requested.status.eq('valid')].reset_index(drop=True)
            if valid.empty:raise RuntimeError('No valid pairs; see rejected_pairs.csv')
        print('Checking known-row features and model identity...',flush=True)
        predictor=None
        if not args.features_only:
            from model_loader import FrozenPredictor
            predictor=FrozenPredictor()
            manifest['model_artifact_hashes']=predictor.hashes
            manifest['calibration']={k:predictor.manifest[k] for k in ['lambda_final','residual_cap','gate_center','gate_scale','high_minus_NCP_threshold']}
        manifest['identity_results']=identity_test(out,predictor)
        if args.identity_only:
            manifest['status']='feature_identity_passed_prediction_pending' if args.features_only else 'identity_passed'
            return 0
        if not args.prepared_run:
            print(f'Calculating features for {len(valid)} valid pairs...',flush=True)
            features=calculate_features(valid)
        finite=np.isfinite(features.to_numpy(float)).all(axis=1)
        if not finite.all():
            bad=valid.loc[~finite].copy();bad['status']='rejected';bad['failure_reason']='nonfinite_feature'
            previous=pd.read_csv(out/'rejected_pairs.csv')
            pd.concat([previous,bad],ignore_index=True).to_csv(out/'rejected_pairs.csv',index=False)
            manifest['number_rejected']+=len(bad)
            valid=valid.loc[finite].reset_index(drop=True);features=features.loc[finite].reset_index(drop=True)
        if valid.empty:raise RuntimeError('No finite feature rows')
        if not args.prepared_run:
            features[PUBLIC_FEATURES].to_csv(out/'feature_matrix.csv',index=False)
            valid[['input_row','pair_id']].assign(feature_row=np.arange(len(valid))).to_csv(out/'feature_row_ids.csv',index=False)
            flags=applicability(features)
            ready=pd.concat([valid,features,flags],axis=1)
            ready.to_csv(out/'prepared_pairs.csv',index=False)
        manifest['number_feature_rows']=len(ready)
        from prepared_io import PREPARED_FILES
        manifest['prepared_output_hashes']={p:sha256(out/p) for p in PREPARED_FILES}
        if args.features_only:
            manifest['status']='features_prepared_prediction_pending'
        else:
            print(f'Applying frozen predictor to {len(valid)} pairs...',flush=True)
            predictions=predictor.predict(features,valid.pair_id)
            result=pd.concat([ready,predictions],axis=1)
            result['status']='predicted'
            ordered=META+PUBLIC_FEATURES+PREDICTION_COLS
            result=result[ordered+[c for c in result.columns if c not in ordered]]
            result.to_csv(out/'predictions.csv',index=False)
            result[MINIMAL].to_csv(out/'predictions_minimal.csv',index=False)
            manifest['number_predicted']=len(result)
            manifest['prediction_columns']=list(result.columns)
            manifest['status']='completed_with_rejections' if manifest['number_rejected'] else 'completed'
        return 0
    except Exception as exc:
        manifest['status']='failed';manifest['error']=str(exc)
        (out/'error_traceback.txt').write_text(traceback.format_exc())
        if not (out/'pipeline_identity_test.txt').exists():
            (out/'pipeline_identity_test.txt').write_text('Identity validation NOT COMPLETED: '+str(exc)+'\nNo new model predictions generated.\n')
        print(f'ERROR: {exc}\nRun audit: {out}',file=sys.stderr)
        return 1
    finally:
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        lines=[f'Run: {name}',f'Status: {manifest["status"]}',f'Requested pairs: {manifest["number_requested_pairs"]}',
               f'Predicted: {manifest["number_predicted"]}',f'Rejected: {manifest["number_rejected"]}',
               'Rejected rows are preserved; duplicate sequence pairs under distinct IDs are retained and flagged.',
               'Predictions are log10(-B22); no back-transformation or model fitting.',
               'Applicability flags do not alter predictions. Histidine features use the fixed historical charge convention, without a new pH correction.']
        if 'error' in manifest:lines.append('Error: '+manifest['error'])
        (out/'qc_summary.txt').write_text('\n'.join(lines)+'\n')
        print(f'Run files: {out}',flush=True)

if __name__=='__main__':sys.exit(main())
