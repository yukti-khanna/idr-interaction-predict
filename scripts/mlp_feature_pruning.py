# mlp_feature_pruning.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector

# -----------------------------
# Config
# -----------------------------
CSV = "sim_pairs_added_features_data_270525.csv"   # <-- change if needed
TARGET = "B22_corr1_log10"
OCN_COL = "Opposite_Charge_Number"
OUT_DIR = "feature_pruning_results"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_FEATURE_COLS = [
    "Opposite_Charge_Number","Average_SCD","SCD_RGG","SCD_TAD","Average_NCPR",
    "Charge_Complementarity","NCPR_RGG","SHD_RGG","Average_Interaction","SHD_TAD",
    "Average_SHD","RGG_length","lambda_RGG","Total_Length","TAD_length",
    "lambda_TAD","NCPR_TAD","Min_lambda","Average_lambda","charge_RGG",
    "Average_charge","charge_TAD"
]

ENGINEERED_COLS = [
    "Normalized_OCN","Charge_Asymmetry","Hydropathy_Diff","Lambda_Ratio"
]

# Your tuned MLP params
MLP_PARAMS = dict(
    activation="tanh",
    alpha=0.01,
    hidden_layer_sizes=(100, 50, 25),
    learning_rate_init=0.01,
    max_iter=500,
    solver="adam",
    random_state=42
)

# Plateau threshold
OCN_PLATEAU = 40.0

# -----------------------------
# Helpers
# -----------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Opposite_Charge_Number" in df and "Total_Length" in df:
        df["Normalized_OCN"] = df["Opposite_Charge_Number"] / (df["Total_Length"].replace(0, np.nan))
    if "charge_TAD" in df and "charge_RGG" in df:
        df["Charge_Asymmetry"] = (df["charge_TAD"] - df["charge_RGG"]).abs()
    if "SHD_TAD" in df and "SHD_RGG" in df:
        df["Hydropathy_Diff"] = df["SHD_TAD"] - df["SHD_RGG"]
    if "lambda_TAD" in df and "lambda_RGG" in df:
        df["Lambda_Ratio"] = df["lambda_TAD"] / (df["lambda_RGG"].replace(0, np.nan) + 1e-6)
    return df

def report_metrics(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"{label}: MSE={mse:.6f}  R2={r2:.4f}")
    return mse, r2

# -----------------------------
# Load + prepare data
# -----------------------------
df = pd.read_csv(CSV)
df = add_engineered_features(df)

ALL_FEATURES = BASE_FEATURE_COLS + ENGINEERED_COLS
df = df.dropna(subset=ALL_FEATURES + [TARGET])

X_full = df[ALL_FEATURES]
y_full = df[TARGET].to_numpy()
ocn_full = df[OCN_COL].to_numpy()

# Hold-out split (keep your seed consistent)
X_train, X_test, y_train, y_test, ocn_train, ocn_test = train_test_split(
    X_full, y_full, ocn_full, test_size=0.25, random_state=42
)

# -----------------------------
# Baseline MLP (all features)
# -----------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(**MLP_PARAMS)),
])

print("\n=== Baseline (all features) ===")
pipe.fit(X_train, y_train)
y_pred_test_all = pipe.predict(X_test)
report_metrics(y_test, y_pred_test_all, "Test (all features)")
mask_plateau_te = ocn_test >= OCN_PLATEAU
if mask_plateau_te.sum() > 0:
    report_metrics(y_test[mask_plateau_te], y_pred_test_all[mask_plateau_te], f"Test (OCN≥{OCN_PLATEAU:.0f})")

# Permutation importance on TEST (informative, not used for selection)
print("\nPermutation importance (test set):")
perm = permutation_importance(pipe, X_test, y_test, n_repeats=20, random_state=42, scoring="r2")
perm_series = pd.Series(perm.importances_mean, index=ALL_FEATURES).sort_values(ascending=False)
perm_series.to_csv(os.path.join(OUT_DIR, "permutation_importance_test.csv"))
print(perm_series.head(10))

# -----------------------------
# Cross-validated feature selection (backward SFS)
# -----------------------------
print("\n=== Cross-validated feature pruning (backward SFS) ===")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# We’ll try several subset sizes and pick the best by CV R²
min_k = max(6, int(len(ALL_FEATURES) * 0.3))  # avoid too tiny sets
max_k = len(ALL_FEATURES)

results = []
best_score = -np.inf
best_k = None
best_support = None

for k in range(min_k, max_k + 1):
    sfs = SequentialFeatureSelector(
        estimator=MLPRegressor(**MLP_PARAMS),
        n_features_to_select=k,
        direction="backward",
        scoring="r2",
        cv=cv,
        n_jobs=-1
    )
    # IMPORTANT: scale inside CV by using a pipeline
    sfs_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("sfs", sfs),
        ("mlp", MLPRegressor(**MLP_PARAMS)),
    ])
    scores = cross_val_score(sfs_pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    mean_r2 = float(np.mean(scores))
    results.append({"k": k, "cv_r2": mean_r2})
    print(f"k={k:2d}  CV R2={mean_r2:.4f}")
    if mean_r2 > best_score:
        best_score = mean_r2
        best_k = k
# -----------------------------
# After the CV loop (best_k & best_score are set)
# -----------------------------
print(f"\nBest k={best_k} by CV R² = {best_score:.4f}")

# 1) Fit a scaler once on TRAIN
scaler_final = StandardScaler().fit(X_train)

# 2) Refit SFS *outside* a pipeline on scaled NumPy arrays (more robust)
Xtr_scaled = scaler_final.transform(X_train).astype(np.float64)
Xte_scaled = scaler_final.transform(X_test).astype(np.float64)

sfs_final = SequentialFeatureSelector(
    estimator=MLPRegressor(**MLP_PARAMS),
    n_features_to_select=best_k,
    direction="backward",
    scoring="r2",
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1
)
sfs_final.fit(Xtr_scaled, y_train)

# 3) Extract selected features by mapping mask back to column names
support_mask = sfs_final.get_support()
selected_features = [f for f, keep in zip(ALL_FEATURES, support_mask) if keep]

print(f"\nSelected {len(selected_features)} features:")
for f in selected_features:
    print(" -", f)

# Save the list
pd.Series(selected_features).to_csv(os.path.join(OUT_DIR, "selected_features.txt"),
                                    index=False, header=False)

# 4) Train a clean pruned pipeline on *unscaled* features (pipeline will scale)
pruned_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(**MLP_PARAMS)),
])
pruned_pipe.fit(X_train[selected_features], y_train)

# 5) Evaluate pruned vs all-features
y_pred_test_pruned = pruned_pipe.predict(X_test[selected_features])
print("\n=== Pruned model (selected features) ===")
mse_p = mean_squared_error(y_test, y_pred_test_pruned)
r2_p  = r2_score(y_test, y_pred_test_pruned)
print(f"Test (pruned) : MSE={mse_p:.6f}  R2={r2_p:.4f}")

mask_plateau_te = ocn_test >= OCN_PLATEAU
if mask_plateau_te.sum() > 0:
    mse_pp = mean_squared_error(y_test[mask_plateau_te], y_pred_test_pruned[mask_plateau_te])
    r2_pp  = r2_score(y_test[mask_plateau_te], y_pred_test_pruned[mask_plateau_te])
    print(f"Test (OCN≥{OCN_PLATEAU:.0f}) pruned: MSE={mse_pp:.6f}  R2={r2_pp:.4f}")

# Compare
mse_all = mean_squared_error(y_test, y_pred_test_all)
r2_all  = r2_score(y_test, y_pred_test_all)
print(f"\nΔR² (pruned - all) on TEST: {r2_p - r2_all:+.4f}")

# Save summary
summary = {
    "best_k": best_k,
    "cv_best_r2": best_score,
    "selected_features": selected_features,
    "test_all": {"mse": mse_all, "r2": r2_all},
    "test_pruned": {"mse": mse_p, "r2": r2_p},
}
import json
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n✅ Results saved in: {OUT_DIR}")
