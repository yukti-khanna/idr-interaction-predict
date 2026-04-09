# compare_hybrid_models.py
import os, argparse, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ======== engineered features ========
ENGINEERED_COLS = [
    "Normalized_OCN", "Charge_Asymmetry", "Hydropathy_Diff", "Lambda_Ratio"
]

BASE_FEATURE_COLS = [
    "Opposite_Charge_Number","Average_SCD","SCD_RGG","SCD_TAD","Average_NCPR",
    "Charge_Complementarity","NCPR_RGG","SHD_RGG","Average_Interaction","SHD_TAD",
    "Average_SHD","RGG_length","lambda_RGG","Total_Length","TAD_length",
    "lambda_TAD","NCPR_TAD","Min_lambda","Average_lambda","charge_RGG",
    "Average_charge","charge_TAD"
]

TARGET_DEFAULT = "B22_corr1_log10"


# ======== helper functions ========
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Opposite_Charge_Number" in df and "Total_Length" in df:
        df["Normalized_OCN"] = df["Opposite_Charge_Number"] / (df["Total_Length"].replace(0, np.nan))
    if "charge_TAD" in df and "charge_RGG" in df:
        df["Charge_Asymmetry"] = np.abs(df["charge_TAD"] - df["charge_RGG"])
    if "SHD_TAD" in df and "SHD_RGG" in df:
        df["Hydropathy_Diff"] = df["SHD_TAD"] - df["SHD_RGG"]
    if "lambda_TAD" in df and "lambda_RGG" in df:
        df["Lambda_Ratio"] = df["lambda_TAD"] / (df["lambda_RGG"].replace(0, np.nan) + 1e-6)
    return df


def resolve_feature_order(bundle, df_after_eng):
    """
    Determine which columns this model expects, creating missing engineered cols if needed.
    """
    scaler = bundle["scaler"]
    if hasattr(scaler, "feature_names_in_"):
        cols = list(scaler.feature_names_in_)
    elif "feature_cols" in bundle:
        cols = list(bundle["feature_cols"])
    else:
        cols = list(BASE_FEATURE_COLS) + list(ENGINEERED_COLS)

    # Add any missing engineered features as neutral 0 values
    for c in cols:
        if c not in df_after_eng.columns:
            print(f"⚠️  Column '{c}' missing in data — filling with 0")
            df_after_eng[c] = 0.0

    # keep only those columns present
    cols = [c for c in cols if c in df_after_eng.columns]
    return cols



def predict_bundle(bundle, df, feature_cols, target_col=TARGET_DEFAULT):
    """Predict base + hybrid for a given model bundle, handling missing features safely."""
    base_mlp   = bundle["base_mlp"]
    scaler     = bundle["scaler"]
    res_model  = bundle["residual_nn"]
    thr        = bundle.get("threshold", 40)       # fallback for older models
    lam        = bundle.get("lambda_hat", 0.6)
    cap        = bundle.get("cap", 0.2)

    Xs = scaler.transform(df[feature_cols])
    y_base = base_mlp.predict(Xs)
    y_corr = res_model.predict(Xs).ravel()
    y_corr = np.clip(y_corr, -cap, cap)

    ocn = df["Opposite_Charge_Number"].to_numpy()
    mask = ocn >= thr
    y_final = y_base.copy()
    y_final[mask] = y_base[mask] + lam * y_corr[mask]

    # metrics if target exists
    if target_col in df.columns:
        y_true = df[target_col].to_numpy()
        mse_b, r2_b = mean_squared_error(y_true, y_base), r2_score(y_true, y_base)
        mse_h, r2_h = mean_squared_error(y_true, y_final), r2_score(y_true, y_final)
        return y_true, y_base, y_final, (mse_b, r2_b, mse_h, r2_h)
    else:
        return None, y_base, y_final, None


# ======== plotting ========
def plot_model_comparison(y_true, preds_dict, ocn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,6))
    for label, preds in preds_dict.items():
        plt.scatter(y_true, preds, s=10, alpha=0.45, label=label)
    lo, hi = min(y_true.min(), min(p.min() for p in preds_dict.values())), \
             max(y_true.max(), max(p.max() for p in preds_dict.values()))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title("Predicted vs Actual — Model Comparison")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_pred_vs_actual.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7,5))
    for label, preds in preds_dict.items():
        plt.scatter(ocn, y_true - preds, s=10, alpha=0.4, label=label)
    plt.axhline(0, color='k', ls='--')
    plt.xlabel("Opposite_Charge_Number"); plt.ylabel("Residual (y - ŷ)")
    plt.title("Residuals vs OCN — Model Comparison")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compare_residuals_vs_ocn.png"), dpi=300)
    plt.close()


# ======== main ========
def main():
    ap = argparse.ArgumentParser(description="Compare two saved MLP+Residual hybrid models")
    ap.add_argument("--model1", required=True, help="Path to first .pkl model (baseline)")
    ap.add_argument("--model2", required=True, help="Path to second .pkl model (new)")
    ap.add_argument("--csv", required=True, help="CSV with features (and optionally target)")
    ap.add_argument("--outdir", default="comparison_results", help="Output directory for plots")
    ap.add_argument("--target", default=TARGET_DEFAULT, help="Target column name")
    args = ap.parse_args()

    # load bundles
    bundle1 = joblib.load(args.model1)
    bundle2 = joblib.load(args.model2)
    print(f"✅ Loaded {args.model1} and {args.model2}")

    # read + engineer
    df = pd.read_csv(args.csv)
    df = add_engineered_features(df)

    feature_cols1 = resolve_feature_order(bundle1, df)
    feature_cols2 = resolve_feature_order(bundle2, df)

    # predict both
    y_true, _, y1, metrics1 = predict_bundle(bundle1, df, feature_cols1, args.target)
    _, _, y2, metrics2 = predict_bundle(bundle2, df, feature_cols2, args.target)

    if y_true is not None:
        mse_b1, r2_b1, mse_h1, r2_h1 = metrics1
        mse_b2, r2_b2, mse_h2, r2_h2 = metrics2
        print("\n=== COMPARISON ===")
        print(f"Old Model:  Base R2={r2_b1:.4f}  Hybrid R2={r2_h1:.4f}")
        print(f"New Model:  Base R2={r2_b2:.4f}  Hybrid R2={r2_h2:.4f}")

    # plotting
    preds_dict = {
        "Old": y1,
        "New": y2
    }
    plot_model_comparison(y_true, preds_dict, df["Opposite_Charge_Number"].to_numpy(), args.outdir)
    print(f"✅ Comparison plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()

