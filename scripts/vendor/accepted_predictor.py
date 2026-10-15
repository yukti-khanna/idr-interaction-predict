"""Unchanged accepted S8 prediction function bodies; inference only."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
MODEL_DIR = None
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def soft_gate(x, center, scale):
    return 1.0 / (1.0 + np.exp(-((np.asarray(x, dtype=float) - center) / scale)))

def predict(df, manifest, base, prep, residual):
    features = manifest["feature_order"]
    missing = [c for c in ["name", *features] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    if df[features].isna().any().any():
        raise RuntimeError("Input feature columns contain missing values")
    X = df[features].to_numpy(float)
    base_pred = np.mean([m.predict(base["scaler"].transform(X)) for m in base["models"]], axis=0)
    raw = residual.predict(prep["scaler"].transform(X), verbose=0).reshape(-1)
    minus = df[manifest["minus_NCP_feature"]].to_numpy(float)
    gate = soft_gate(minus, manifest["gate_center"], manifest["gate_scale"])
    uncapped = manifest["lambda_final"] * gate * raw
    applied = manifest["lambda_final"] * gate * np.clip(raw, -manifest["residual_cap"], manifest["residual_cap"])
    out = df.copy()
    out["base_prediction"] = base_pred
    out["raw_residual_prediction"] = raw
    out["gate_value"] = gate
    out["lambda_final"] = manifest["lambda_final"]
    out["uncapped_correction"] = uncapped
    out["applied_correction"] = applied
    out["final_hybrid_prediction"] = base_pred + applied
    out["prediction_identity_error"] = np.abs(out["final_hybrid_prediction"] - (out["base_prediction"] + out["applied_correction"]))
    out["model_manifest_sha256"] = sha(MODEL_DIR / "MODEL_MANIFEST.json")
    out["base_model_sha256"] = sha(MODEL_DIR / "base_model_ensemble.joblib")
    out["residual_preprocessor_sha256"] = sha(MODEL_DIR / "residual_preprocessor.joblib")
    out["residual_model_sha256"] = sha(MODEL_DIR / "residual_nn.keras")
    if "B22_corr1_log10" in out.columns:
        out["residual"] = out["B22_corr1_log10"] - out["final_hybrid_prediction"]
        out["absolute_error"] = np.abs(out["residual"])
    return out
