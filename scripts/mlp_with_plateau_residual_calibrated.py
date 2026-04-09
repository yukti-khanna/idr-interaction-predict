# mlp_with_plateau_residual_calibrated.py
# -------------------------------------------------------------
# Full pipeline: Base MLP + calibrated plateau residual NN
# -------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

import tensorflow as tf
from tensorflow.keras import layers as L, callbacks as C, optimizers as O, regularizers as R


# =============================================================
# 1) Base MLP (same as your optimised version)
# =============================================================
def mlp_optimised(X_train, X_test, y_train, y_test):
    y_train_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
    y_test_arr  = y_test.to_numpy()  if hasattr(y_test,  "to_numpy")  else np.asarray(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    best_params = {
        'activation': 'tanh',
        'alpha': 0.01,
        'hidden_layer_sizes': (128, 64, 32, 16),
        'learning_rate_init': 0.01,
        'max_iter': 500,
        'solver': 'adam',
        'random_state': 42
    }

    mlp = MLPRegressor(**best_params)

    # --- CV ---
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores, r2_scores = [], []
    for tr_idx, val_idx in cv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train_arr[tr_idx], y_train_arr[val_idx]
        mlp.fit(X_tr, y_tr)
        y_pred_val = mlp.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, y_pred_val))
        r2_scores.append(r2_score(y_val, y_pred_val))

    avg_mse, avg_r2 = np.mean(mse_scores), np.mean(r2_scores)
    print(f"[Base MLP] CV  : MSE={avg_mse:.5f}  R2={avg_r2:.4f}")

    # --- Full fit ---
    mlp.fit(X_train_scaled, y_train_arr)
    y_pred_test = mlp.predict(X_test_scaled)
    mse_test = mean_squared_error(y_test_arr, y_pred_test)
    r2_test  = r2_score(y_test_arr, y_pred_test)
    print(f"[Base MLP] Test: MSE={mse_test:.5f}  R2={r2_test:.4f}")

    return mlp, scaler, y_pred_test

from sklearn.utils import shuffle

def mlp_ensemble(X_train, X_test, y_train, y_test, n_models=5):
    """
    Train n_models independent MLPs with different seeds and average predictions.
    Improves stability and often boosts R² by ~0.01–0.02.
    """
    preds_test = []
    seeds = np.arange(n_models)
    scalers = []
    
    for seed in seeds:
        print(f"\n[Ensemble model {seed+1}/{n_models}]")
        X_tr, y_tr = shuffle(X_train, y_train, random_state=seed)
        mlp, scaler, y_pred_test = mlp_optimised(X_tr, X_test, y_tr, y_test)
        scalers.append(scaler)
        preds_test.append(y_pred_test)
    y_pred_mean = np.mean(preds_test, axis=0)
    mse = mean_squared_error(y_test, y_pred_mean)
    r2  = r2_score(y_test, y_pred_mean)
    print(f"\n[Ensemble mean] MSE={mse:.5f}  R2={r2:.4f}")
    return y_pred_mean, scalers[0]

# =============================================================
# 2) Small residual NN for plateau corrections
# =============================================================
def _build_residual_nn(input_dim: int):
    inp = L.Input(shape=(input_dim,))
    h = L.Dense(16, activation="tanh", kernel_regularizer=R.l2(5e-4))(inp)
    h = L.Dense(8,  activation="tanh", kernel_regularizer=R.l2(5e-4))(h)
    out = L.Dense(1)(h)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=O.Adam(learning_rate=2e-4),
        loss=tf.keras.losses.Huber(delta=0.75),
        metrics=["mae"]
    )
    return model


# =============================================================
# 3) Train MLP + plateau residual NN + calibration
# =============================================================
def train_mlp_with_plateau_residual(
    X_train, X_test, y_train, y_test,
    ocn_train, ocn_test,
    thr_quantile=0.90,
    y_pred_test_base=None,    # optional: use ensemble preds
    base_mlp=None,            # optional: pass pretrained model
    scaler=None               # optional: pass existing scaler
):
    """
    Train or use a precomputed base MLP model, then fit a residual NN to plateau region.

    If y_pred_test_base, base_mlp, and scaler are all provided, the function skips
    base training and uses those. Otherwise, it trains a new MLP internally.
    """

    # === 1) Ensure we have a base model and predictions ===
    if base_mlp is None or scaler is None or y_pred_test_base is None:
        print("[INFO] Training new base MLP...")
        base_mlp, scaler, y_pred_test_base = mlp_optimised(X_train, X_test, y_train, y_test)
    else:
        print("[INFO] Using provided base predictions and scaler (ensemble or pretrained).")

    # === 2) Prepare scaled data ===
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)
    ytr = np.asarray(y_train)
    yte = np.asarray(y_test)

    # --- Base train predictions ---
    if base_mlp is not None:
        y_pred_tr_base = base_mlp.predict(Xtr_s)
    else:
        # if ensemble (no base_mlp given), approximate by fitting a Ridge on (X_train, y_pred_test_base)
        print("[WARN] base_mlp is None — fitting lightweight surrogate to compute train preds.")
        from sklearn.linear_model import Ridge
        surrogate = Ridge(alpha=0.1)
        surrogate.fit(X_train, y_train)
        y_pred_tr_base = surrogate.predict(Xtr_s)

    train_residuals = ytr - y_pred_tr_base

    # === 3) Define plateau ===
    thr = np.quantile(ocn_train, thr_quantile)
    mask_plateau_tr = ocn_train >= thr
    print(f"[Residual NN] Plateau threshold q{int(thr_quantile*100)} = {thr:.3f}  |  tail n = {mask_plateau_tr.sum()}")

    # === 4) Fit residual NN ===
    res_model = _build_residual_nn(Xtr_s.shape[1])
    Xtr_plateau = Xtr_s[mask_plateau_tr]
    res_plateau = train_residuals[mask_plateau_tr]

    bs = 8 if len(res_plateau) < 200 else 32
    es  = C.EarlyStopping(patience=30, restore_best_weights=True, monitor="val_loss")
    rlr = C.ReduceLROnPlateau(patience=10, factor=0.5, monitor="val_loss")

    gate_tr = soft_gate(ocn_train, center=40.0, scale=12.0)
    sample_w = gate_tr[mask_plateau_tr]  # weights for tail subset
    

    if len(res_plateau) >= 12:
        res_model.fit(Xtr_plateau, res_plateau,
                      validation_split=0.2, epochs=400, batch_size=bs,
                      sample_weight=sample_w, callbacks=[es, rlr], verbose=0)
    else:
        res_model.fit(Xtr_s, train_residuals,
                      validation_split=0.2, epochs=300, batch_size=32,
                      csample_weight=sample_w, allbacks=[es, rlr], verbose=0)

    # === 5) Calibrate λ (shrinkage) and cap ===
    tail_idx = np.where(ocn_train >= thr)[0]
    idx_fit, idx_cal = train_test_split(tail_idx, test_size=0.35, random_state=42)
    corr_tr_all = res_model.predict(Xtr_s, verbose=0).ravel()
    base_tr_all = y_pred_tr_base
    r_true_cal  = (ytr - base_tr_all)[idx_cal]
    r_hat_cal   = corr_tr_all[idx_cal]
    num = np.dot(r_true_cal, r_hat_cal)
    den = np.dot(r_hat_cal, r_hat_cal) + 1e-12
    lambda_hat = np.clip(float(num / den), 0.0, 0.6)
    cap = np.percentile(np.abs(r_true_cal), 85)
    print(f"[Calibration] lambda_hat = {lambda_hat:.3f}, cap = {cap:.3f}")

    # === 6) Apply calibrated correction on TEST ===
    corr_te_all = res_model.predict(Xte_s, verbose=0).ravel()
    gate_te = soft_gate(ocn_test, center=40.0, scale=12.0)
    corr_te_all = np.clip(corr_te_all, -cap, cap)

    mask_plateau_te = ocn_test >= thr

    y_pred_test_hybrid = y_pred_test_base.copy()
    y_pred_test_hybrid[mask_plateau_te] = (
        y_pred_test_base[mask_plateau_te] + lambda_hat * corr_te_all[mask_plateau_te]
    )
    y_pred_test_hybrid = y_pred_test_base + (lambda_hat * gate_te * corr_te_all)
    return {
        "base_model": base_mlp,
        "scaler": scaler,
        "residual_model": res_model,
        "threshold": thr,
        "lambda_hat": lambda_hat,
        "cap": cap,
        "y_pred_test_base": y_pred_test_base,
        "y_pred_test_hybrid": y_pred_test_hybrid
    }

    # --- Metrics ---
    def _report(y_true, y_pred, label):
        mse, r2 = mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)
        print(f"{label}: MSE={mse:.6f}  R2={r2:.4f}")
        return mse, r2

    print("\n=== TEST METRICS ===")
    _report(yte, y_pred_test_base,   "Base MLP")
    _report(yte, y_pred_test_hybrid, "Hybrid (calibrated)")
    low_te, high_te = ~mask_plateau_te, mask_plateau_te
    if low_te.sum():  _report(yte[low_te],  y_pred_test_hybrid[low_te],  "Hybrid LOW")
    if high_te.sum(): _report(yte[high_te], y_pred_test_hybrid[high_te], "Hybrid HIGH")

    return {
        "base_model": base_mlp,
        "scaler": scaler,
        "residual_model": res_model,
        "threshold": thr,
        "lambda_hat": lambda_hat,
        "cap": cap,
        "y_pred_test_base": y_pred_test_base,
        "y_pred_test_hybrid": y_pred_test_hybrid
    }


# =============================================================
# 4) Held-out evaluation
# =============================================================
def eval_on_heldout(results, heldout_csv, feature_cols,
                    target_col="B22_corr1_log10",
                    ocn_col="Opposite_Charge_Number",
                    out_dir="mlp_plateau_residual_results"):
    os.makedirs(out_dir, exist_ok=True)
    base_mlp   = results["base_model"]
    scaler     = results["scaler"]
    res_model  = results["residual_model"]
    thr        = results["threshold"]
    lam        = results["lambda_hat"]
    cap        = results["cap"]

    # 1) Read + add engineered features first
    df_h = add_engineered_features(pd.read_csv(heldout_csv))

    # 2) Use a local copy of the feature list (don’t mutate caller)
    full_cols = list(feature_cols)   # assume this already includes ENGINEERED_COLS

    # 3) Now it’s safe to drop rows with missing values
    df_h = df_h.dropna(subset=full_cols + [target_col])

    # 4) Align columns and transform
    X_h  = df_h[full_cols].to_numpy()
    y_h  = df_h[target_col].to_numpy()
    ocn_h = df_h[ocn_col].to_numpy()
    X_hs = scaler.transform(X_h)

    yhat_base_h = base_mlp.predict(X_hs)
    ycorr_all_h = res_model.predict(X_hs, verbose=0).ravel()
    gate_h = soft_gate(ocn_h, center=40.0, scale=12.0)
    ycorr_all_h = np.clip(ycorr_all_h, -cap, cap)
    
    mask_plateau_h = ocn_h >= thr
    yhat_hyb_h = yhat_base_h.copy()
    yhat_hyb_h[mask_plateau_h] = (
        yhat_base_h[mask_plateau_h] + lam * ycorr_all_h[mask_plateau_h]
    )
    yhat_hyb_h  = yhat_base_h + (lam * gate_h * ycorr_all_h)

    def _report(y_true, y_pred, label):
        mse, r2 = mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)
        print(f"{label}: MSE={mse:.6f}  R2={r2:.4f}")
        return mse, r2

    print("\n=== HELD-OUT METRICS ===")
    _report(y_h, yhat_base_h, "Base MLP (held-out)")
    _report(y_h, yhat_hyb_h,  "Hybrid (held-out)")

    # --- Visuals ---
    plt.figure(figsize=(6,6))
    plt.scatter(y_h, yhat_base_h, s=10, alpha=0.4, label="Base")
    plt.scatter(y_h, yhat_hyb_h,  s=10, alpha=0.4, label="Hybrid")
    lo, hi = min(y_h.min(), yhat_hyb_h.min()), max(y_h.max(), yhat_hyb_h.max())
    plt.plot([lo, hi], [lo, hi], 'k--', lw=1)
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title("Predicted vs Actual — Held-out")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pred_vs_actual_heldout.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.scatter(ocn_h, y_h - yhat_base_h, s=10, alpha=0.4, label="Base")
    plt.scatter(ocn_h, y_h - yhat_hyb_h,  s=10, alpha=0.4, label="Hybrid")
    plt.axhline(0, color='k', ls='--')
    plt.xlabel("Opposite_Charge_Number"); plt.ylabel("Residual (y - ŷ)")
    plt.title("Residuals vs OCN — Held-out")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residuals_vs_ocn_heldout.png"), dpi=300)
    plt.close()

    print(f"✅ Held-out figures saved to: {out_dir}")
    return {"y_true": y_h, "y_base": yhat_base_h, "y_hybrid": yhat_hyb_h}

ENGINEERED_COLS = [
    "Normalized_OCN", "Charge_Asymmetry", "Hydropathy_Diff", "Lambda_Ratio"
]

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # (a) Normalized OCN
    df["Normalized_OCN"] = df["Opposite_Charge_Number"] / df["Total_Length"]
    # (b) Charge asymmetry
    df["Charge_Asymmetry"] = np.abs(df["charge_TAD"] - df["charge_RGG"])
    # (c) Hydropathy difference
    df["Hydropathy_Diff"] = df["SHD_TAD"] - df["SHD_RGG"]
    # (d) Lambda ratio
    df["Lambda_Ratio"] = df["lambda_TAD"] / (df["lambda_RGG"] + 1e-6)
    return df

def soft_gate(ocn_vec, center=40.0, scale=12.0):
    z = (ocn_vec - center) / scale
    return 1.0 / (1.0 + np.exp(-z))

# =============================================================
# 5) Run end-to-end
# =============================================================
if __name__ == "__main__":
    OUT_DIR = "mlp_plateau_residual_results"
    os.makedirs(OUT_DIR, exist_ok=True)

    CSV_TRAIN = "sim_pairs_added_features_data_270525.csv"
    CSV_HELD  = "taken_out_features_data_merged.csv"
    TARGET = "B22_corr1_log10"
    FEATURE_COLS = [
        "Opposite_Charge_Number","Average_SCD","SCD_RGG","SCD_TAD","Average_NCPR",
        "Charge_Complementarity","NCPR_RGG","SHD_RGG","Average_Interaction","SHD_TAD",
        "Average_SHD","RGG_length","lambda_RGG","Total_Length","TAD_length",
        "lambda_TAD","NCPR_TAD","Min_lambda","Average_lambda","charge_RGG",
        "Average_charge","charge_TAD"
    ]
    Selected_COLS = [
    "Opposite_Charge_Number",
    "Normalized_OCN",
    "charge_TAD",
    "Average_SHD",
    "lambda_RGG",
    "Lambda_Ratio",
    "lambda_TAD",
    "TAD_length",
    "Average_Interaction",
    "SHD_TAD"
    ]


    # full feature list used for training (order matters!)
    FEATURE_COLS_FULL = FEATURE_COLS + ENGINEERED_COLS

    # load TRAIN with engineered features, then dropna on full set
    df = add_engineered_features(pd.read_csv(CSV_TRAIN))
    df = df.dropna(subset=Selected_COLS + [TARGET])

    X = df[Selected_COLS]
    y = df[TARGET]
    ocn = df["Opposite_Charge_Number"].to_numpy()

    X_train, X_test, y_train, y_test, ocn_train, ocn_test = train_test_split(
        X, y, ocn, test_size=0.25, random_state=42
    )

    y_pred_test_ens, my_scaler = mlp_ensemble(X_train, X_test, y_train, y_test, n_models=5)

    results = train_mlp_with_plateau_residual(
        X_train, X_test, y_train, y_test,
        ocn_train=ocn_train,
        ocn_test=ocn_test,
        thr_quantile=0.90,
        y_pred_test_base=y_pred_test_ens,
        scaler = my_scaler
    )


    heldout_results = eval_on_heldout(
        results=results,
        heldout_csv=CSV_HELD,
        feature_cols=Selected_COLS,   # <- pass the full list, not the base list
        target_col=TARGET,
        ocn_col="Opposite_Charge_Number",
        out_dir=OUT_DIR
    )

    import joblib, os

# Define output folder and filename
OUT_PATH = os.path.join(OUT_DIR, "mlp_plateau_calibrated_bundle_sel_cols.pkl")

bundle = {
    "base_mlp": results["base_model"],
    "scaler": results["scaler"],
    "residual_nn": results["residual_model"],
    "threshold": results["threshold"],
    "lambda_hat": results["lambda_hat"],
    "cap": results["cap"]
}

joblib.dump(bundle, OUT_PATH)
print(f"✅ Model bundle saved successfully to:\n{OUT_PATH}")
