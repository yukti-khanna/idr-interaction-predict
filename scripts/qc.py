"""Applicability flags and mandatory known-row comparisons."""
import json
import numpy as np
import pandas as pd
from configuration import PIPELINE_ROOT, PUBLIC_FEATURES
from feature_engineering import calculate_features

# Strict deterministic checks plus explicit float32 TensorFlow backend allowance.
TOLERANCES={'features':1e-10,'base_prediction':1e-10,'raw_residual_prediction':1e-7,
            'residual_gate':1e-10,'applied_correction':1e-8,'final_hybrid_prediction':1e-8}

def applicability(features):
    ranges=pd.read_csv(PIPELINE_ROOT/'resources/development_feature_ranges.csv').set_index('feature')
    flags=pd.DataFrame(index=features.index)
    outside=[]; robust=[]; warnings=[]
    for _, row in features.iterrows():
        outside.append([c for c in PUBLIC_FEATURES if row[c]<ranges.loc[c,'min'] or row[c]>ranges.loc[c,'max']])
        robust.append([c for c in PUBLIC_FEATURES if row[c]<ranges.loc[c,'p005'] or row[c]>ranges.loc[c,'p995']])
        messages=[]
        for c,lo,hi in [('mean_lambda_RGG',.4,.72),('mean_lambda_AHD',.31,.50)]:
            if not lo<=row[c]<=hi: messages.append(c+':outside_observed_control_range')
            if not 0<=row[c]<=1: messages.append(c+':outside_0_1_check_parameter_scale')
        warnings.append(';'.join(messages))
    flags['high_NCP']=features.minus_NCP>=40
    flags['n_features_outside_development_range']=[len(x) for x in outside]
    flags['features_outside_development_range']=[';'.join(x) for x in outside]
    flags['n_features_outside_development_robust_range']=[len(x) for x in robust]
    flags['features_outside_development_robust_range']=[';'.join(x) for x in robust]
    flags['feature_warnings']=warnings
    return flags

def identity_test(output, predictor=None):
    ref=pd.read_csv(PIPELINE_ROOT/'resources/identity_reference.csv')
    new=calculate_features(ref)
    reports=[]
    for col in PUBLIC_FEATURES:
        diff=np.abs(new[col].to_numpy()-ref[col].to_numpy())
        reports.append({'quantity':col,'n':len(ref),'max_absolute_difference':float(diff.max()),'tolerance':TOLERANCES['features'],'passed':bool(np.isfinite(diff).all() and (diff<=TOLERANCES['features']).all())})
    feature_pass=all(x['passed'] for x in reports)
    # Never feed mismatched regenerated features to the residual model.
    if predictor is not None and feature_pass:
        predicted=predictor.predict(new,ref.pair_id)
        for col in ['base_prediction','raw_residual_prediction','residual_gate','applied_correction','final_hybrid_prediction']:
            stored='gate_value' if col=='residual_gate' else col
            diff=np.abs(predicted[col].to_numpy()-ref[stored].to_numpy())
            reports.append({'quantity':col,'n':len(ref),'max_absolute_difference':float(diff.max()),'tolerance':TOLERANCES[col],'passed':bool(np.isfinite(diff).all() and (diff<=TOLERANCES[col]).all())})
    passed=all(x['passed'] for x in reports)
    pd.DataFrame(reports).to_csv(output/'pipeline_identity_test.csv',index=False)
    text=['Known-row identity test',f'rows={len(ref)}; ordinary=10; high_negative_net_charge_product=10',
          f'feature_identity_passed={feature_pass}',f'prediction_identity={"PASSED" if predictor is not None and passed else "FAILED" if predictor is not None else "NOT RUN (feature-only preparation)"}',
          'rtol=0 for all comparisons; feature/base/gate atol=1e-10; TensorFlow raw atol=1e-7; applied/final atol=1e-8.',
          'Backend allowance is explicit: accepted S8 CPU runs previously differed by 4.89e-8 raw and 1.62e-9 final. No model changes are allowed.']
    text += [f'{x["quantity"]}: max_diff={x["max_absolute_difference"]:.17g}; tolerance={x["tolerance"]}; passed={x["passed"]}' for x in reports]
    (output/'pipeline_identity_test.txt').write_text('\n'.join(text)+'\n')
    if not passed: raise RuntimeError('Known-row identity check failed; see pipeline_identity_test.csv. New predictions were not generated.')
    return reports
