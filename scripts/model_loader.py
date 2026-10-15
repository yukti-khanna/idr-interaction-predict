"""Load the authoritative server model and reuse the accepted S8 predictor."""
import json
import numpy as np
from configuration import PIPELINE_ROOT, MODEL_DIR, SCHEMA, PUBLIC_FEATURES, INTERNAL_FEATURES, PUBLIC_TO_INTERNAL, sha256
from vendor import accepted_predictor

class FrozenPredictor:
    def __init__(self):
        path=MODEL_DIR/'MODEL_MANIFEST.json'
        actual=sha256(path)
        if actual!=SCHEMA['expected_manifest_sha256']:
            raise RuntimeError(f'Model manifest identity mismatch: expected {SCHEMA["expected_manifest_sha256"]}, got {actual}. Use the authoritative accepted model in {MODEL_DIR}; no model was substituted.')
        self.manifest=json.loads(path.read_text())
        if self.manifest['feature_order']!=INTERNAL_FEATURES:
            raise RuntimeError('Stored feature order does not match the final schema')
        expected_hashes=json.loads((PIPELINE_ROOT/'config/accepted_model_hashes.json').read_text())
        self.hashes={}
        for name in ['MODEL_MANIFEST.json','base_model_ensemble.joblib','residual_preprocessor.joblib','residual_nn.keras']:
            artifact=MODEL_DIR/name
            if not artifact.is_file():
                raise RuntimeError(f'Missing authoritative model file: {artifact}. No conversion or substitution is attempted.')
            self.hashes[name]=sha256(artifact)
            if self.hashes[name]!=expected_hashes[name]:
                raise RuntimeError(f'Authoritative artifact hash mismatch for {name}: expected {expected_hashes[name]}, got {self.hashes[name]}')
        import joblib
        import tensorflow as tf
        self.base=joblib.load(MODEL_DIR/'base_model_ensemble.joblib')
        self.prep=joblib.load(MODEL_DIR/'residual_preprocessor.joblib')
        self.residual=tf.keras.models.load_model(MODEL_DIR/'residual_nn.keras',compile=False)
        if len(self.base['models'])!=5: raise RuntimeError('Expected five frozen base ensemble members')
        for obj in [self.base,self.prep]:
            if 'feature_order' in obj and list(obj['feature_order'])!=INTERNAL_FEATURES:
                raise RuntimeError('Stored preprocessor feature order mismatch')
        accepted_predictor.MODEL_DIR=MODEL_DIR

    def predict(self, features, pair_ids):
        if not np.isfinite(features[PUBLIC_FEATURES].to_numpy(float)).all():
            raise RuntimeError('Non-finite model inputs')
        internal=features[PUBLIC_FEATURES].rename(columns=PUBLIC_TO_INTERNAL).copy()
        internal.insert(0,'name',list(pair_ids))
        out=accepted_predictor.predict(internal,self.manifest,self.base,self.prep,self.residual)
        Xb=self.base['scaler'].transform(internal[INTERNAL_FEATURES].to_numpy(float))
        members=np.column_stack([model.predict(Xb) for model in self.base['models']])
        if not np.allclose(members.mean(axis=1),out.base_prediction,atol=1e-12,rtol=0):
            raise RuntimeError('Member mean differs from accepted base prediction')
        selected=out[['base_prediction','raw_residual_prediction','gate_value','uncapped_correction','applied_correction','final_hybrid_prediction']].rename(columns={'gate_value':'residual_gate','uncapped_correction':'weighted_residual'}).copy()
        for i in range(5): selected[f'base_model_{i+1}']=members[:,i]
        selected['lambda_final']=self.manifest['lambda_final']
        selected['residual_cap']=self.manifest['residual_cap']
        if not np.isfinite(selected.to_numpy(float)).all(): raise RuntimeError('Non-finite prediction')
        if not np.allclose(selected.final_hybrid_prediction,selected.base_prediction+selected.applied_correction,rtol=0,atol=1e-12):
            raise RuntimeError('Final prediction does not equal base plus correction')
        selected.index=features.index
        return selected
