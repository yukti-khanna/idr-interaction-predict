#!/usr/bin/env python
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reviewer_hybrid_residual_ablation_F2_training_only"
INFILE = ROOT / "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
HIST = Path("/Users/yuktikhanna/idr-interaction-predict/scripts/mlp_with_plateau_residual_calibrated.py")
TFPY = Path("/Users/yuktikhanna/miniconda3/envs/tensorflow/bin/python")
TARGET = "B22_corr1_log10"
FEATURES = ["Opposite_Charge_Number", "mean_lambda_RGG", "mean_lambda_TAD", "SHD_RGG", "SHD_TAD"]
SEEDS = [0, 1, 2, 3, 4]
OCN_HIGH = 40


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def allowed_existing_files():
    sealed = {"taken_out_preds_with_hybrid_021125_sel_cols.csv", "unsim_preds_with_hybrid_021125_sel_cols.csv"}
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or OUT in p.parents or p.name in sealed:
            continue
        files.append(p)
    return files


def metrics(y, pred):
    return {
        "n": int(len(y)),
        "pearson_r": float(stats.pearsonr(y, pred).statistic) if len(y) > 2 else np.nan,
        "spearman_rho": float(stats.spearmanr(y, pred).statistic) if len(y) > 2 else np.nan,
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "mae": float(mean_absolute_error(y, pred)),
    }


def ensemble_predict(x_train, y_train, x_pred):
    scaler = StandardScaler()
    xt = scaler.fit_transform(x_train)
    xp = scaler.transform(x_pred)
    preds = []
    for seed in SEEDS:
        model = MLPRegressor(
            activation="tanh", alpha=0.01, hidden_layer_sizes=(128, 64, 32, 16),
            learning_rate_init=0.01, max_iter=500, solver="adam", random_state=seed
        )
        model.fit(xt, y_train)
        preds.append(model.predict(xp))
    return np.mean(np.vstack(preds), axis=0)


def residual_nn_predict(x_fit, y_fit, x_pred, work_prefix):
    fit_path = OUT / f"{work_prefix}_fit.npz"
    pred_path = OUT / f"{work_prefix}_pred.npy"
    np.savez(fit_path, x_fit=x_fit, y_fit=y_fit, x_pred=x_pred)
    code = r"""
import os, sys, numpy as np, tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
fit_path, pred_path = sys.argv[1], sys.argv[2]
d = np.load(fit_path)
tf.keras.utils.set_random_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(d["x_fit"].shape[1],)),
    tf.keras.layers.Dense(16, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
    tf.keras.layers.Dense(8, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
              loss=tf.keras.losses.Huber(delta=0.75))
model.fit(d["x_fit"], d["y_fit"], epochs=400, batch_size=min(32, len(d["y_fit"])), verbose=0)
pred = model.predict(d["x_pred"], verbose=0).reshape(-1)
np.save(pred_path, pred)
"""
    subprocess.run([str(TFPY), "-c", code, str(fit_path), str(pred_path)], check=True)
    out = np.load(pred_path)
    fit_path.unlink(missing_ok=True)
    pred_path.unlink(missing_ok=True)
    return out


def gate(ocn, center=40, scale=12):
    return 1 / (1 + np.exp(-((ocn - center) / scale)))


def bias_summary(y, base, ocn):
    residual = y - base
    low = ocn < OCN_HIGH
    high = ocn >= OCN_HIGH
    return {
        "n_low_ocn": int(low.sum()),
        "n_high_ocn": int(high.sum()),
        "mean_residual_low_ocn": float(residual[low].mean()),
        "median_residual_low_ocn": float(np.median(residual[low])),
        "mean_residual_high_ocn": float(residual[high].mean()) if high.any() else np.nan,
        "median_residual_high_ocn": float(np.median(residual[high])) if high.any() else np.nan,
        "rmse_low_ocn": float(np.sqrt(np.mean(residual[low] ** 2))),
        "rmse_high_ocn": float(np.sqrt(np.mean(residual[high] ** 2))) if high.any() else np.nan,
        "pearson_residual_ocn": float(stats.pearsonr(residual, ocn).statistic),
        "spearman_residual_ocn": float(stats.spearmanr(residual, ocn).statistic),
    }


def main():
    OUT.mkdir(exist_ok=True)
    before = {str(p.relative_to(ROOT)): sha256(p) for p in allowed_existing_files()}
    df = pd.read_csv(INFILE)
    df["row_id"] = np.arange(len(df))
    df["mean_lambda_RGG"] = df["lambda_RGG"] / df["RGG_length"]
    df["mean_lambda_TAD"] = df["lambda_TAD"] / df["TAD_length"]
    need = [TARGET, "rgg", "tad"] + FEATURES
    d = df.dropna(subset=need).copy().reset_index(drop=True)
    x = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    ocn = d["Opposite_Charge_Number"].to_numpy(float)

    outer = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_rows, inner_rows, split_rows, param_rows, pred_rows, fold_rows, bias_rows = [], [], [], [], [], [], []
    for outer_fold, (tr, te) in enumerate(outer.split(d), start=1):
        for idx in te:
            outer_rows.append({"outer_fold": outer_fold, "row_id": int(d.loc[idx, "row_id"])})
        inner_pred = np.empty(len(tr))
        inner = KFold(n_splits=5, shuffle=True, random_state=42)
        for inner_fold, (itr_local, iva_local) in enumerate(inner.split(tr), start=1):
            itr, iva = tr[itr_local], tr[iva_local]
            inner_pred[iva_local] = ensemble_predict(x[itr], y[itr], x[iva])
            for row in iva:
                inner_rows.append({"outer_fold": outer_fold, "inner_fold": inner_fold, "row_id": int(d.loc[row, "row_id"]),
                                   "observed": y[row], "inner_oof_base_prediction": inner_pred[np.where(tr == row)[0][0]],
                                   "inner_oof_residual": y[row] - inner_pred[np.where(tr == row)[0][0]],
                                   "Opposite_Charge_Number": ocn[row]})
        inner_res = y[tr] - inner_pred
        bias_rows.append({"outer_fold": outer_fold, **bias_summary(y[tr], inner_pred, ocn[tr])})

        high_local = np.where(ocn[tr] >= OCN_HIGH)[0]
        fit_local, cal_local = train_test_split(high_local, train_size=0.65, random_state=42, shuffle=True)
        for local, split in [(i, "residual_fit") for i in fit_local] + [(i, "calibration") for i in cal_local]:
            split_rows.append({"outer_fold": outer_fold, "row_id": int(d.loc[tr[local], "row_id"]), "split": split})

        rscaler = StandardScaler()
        rscaler.fit(x[tr][fit_local])
        all_train_scaled = rscaler.transform(x[tr])
        test_scaled = rscaler.transform(x[te])
        raw_all = residual_nn_predict(
            all_train_scaled[fit_local],
            inner_res[fit_local],
            np.vstack([all_train_scaled, test_scaled]),
            f"outer{outer_fold}",
        )
        raw_train = raw_all[:len(tr)]
        raw_test = raw_all[len(tr):]
        cal_raw = raw_train[cal_local]
        cal_y = inner_res[cal_local]
        denom = float(np.dot(cal_raw, cal_raw))
        unconstrained = float(np.dot(cal_raw, cal_y) / denom) if denom > 0 else 0.0
        lambda_hat = float(np.clip(unconstrained, 0, 0.6))
        caps = {pct: float(np.percentile(np.abs(cal_y), pct)) for pct in [75, 85, 95]}
        cap = caps[85]

        base_test = ensemble_predict(x[tr], y[tr], x[te])
        clipped = np.clip(raw_test, -cap, cap)
        g = gate(ocn[te], 40, 12)
        corr = lambda_hat * g * clipped
        hybrid = base_test + corr
        for j, idx in enumerate(te):
            pred_rows.append({
                "outer_fold": outer_fold, "row_id": int(d.loc[idx, "row_id"]), "rgg": d.loc[idx, "rgg"], "tad": d.loc[idx, "tad"],
                "observed": y[idx], "Opposite_Charge_Number": ocn[idx], "base_prediction": base_test[j],
                "raw_residual_prediction": raw_test[j], "clipped_residual_prediction": clipped[j], "gate_value": g[j],
                "lambda_hat": lambda_hat, "cap_75": caps[75], "cap_85": caps[85], "cap_95": caps[95],
                "applied_correction": corr[j], "hybrid_prediction": hybrid[j],
            })
        mb, mh = metrics(y[te], base_test), metrics(y[te], hybrid)
        fold_rows.append({"outer_fold": outer_fold, "base_r2": mb["r2"], "hybrid_r2": mh["r2"], "delta_r2": mh["r2"] - mb["r2"],
                          "base_rmse": mb["rmse"], "hybrid_rmse": mh["rmse"], "delta_rmse": mh["rmse"] - mb["rmse"],
                          "base_mae": mb["mae"], "hybrid_mae": mh["mae"], "delta_mae": mh["mae"] - mb["mae"],
                          "unconstrained_lambda": unconstrained, "lambda_hat": lambda_hat, "cap": cap,
                          "residual_fit_n": len(fit_local), "calibration_n": len(cal_local),
                          "high_ocn_outer_test_n": int((ocn[te] >= OCN_HIGH).sum())})
        param_rows.append({"outer_fold": outer_fold, "gate_center": 40, "gate_scale": 12, "lambda_upper": 0.6,
                           "unconstrained_lambda": unconstrained, "lambda_hat": lambda_hat, "cap_percentile": 85,
                           "cap": cap, "cap_75": caps[75], "cap_85": caps[85], "cap_95": caps[95],
                           "residual_fit_n": len(fit_local), "calibration_n": len(cal_local)})

    pd.DataFrame(outer_rows).to_csv(OUT / "outer_fold_assignments.csv", index=False)
    inner_df = pd.DataFrame(inner_rows)
    inner_df.to_csv(OUT / "inner_oof_residuals.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "residual_training_splits.csv", index=False)
    pd.DataFrame(param_rows).to_csv(OUT / "outer_fold_parameters.csv", index=False)
    preds = pd.DataFrame(pred_rows).sort_values("row_id")
    preds.to_csv(OUT / "outer_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "foldwise_base_vs_hybrid.csv", index=False)
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(OUT / "residual_bias_by_outer_fold.csv", index=False)
    bias.describe().T.to_csv(OUT / "residual_bias_summary.csv")

    write_metrics(preds)
    sensitivity(preds)
    historical_notes()
    figures(preds, inner_df)
    write_text(preds, pd.DataFrame(fold_rows), bias, before)


def write_metrics(preds):
    y = preds["observed"].to_numpy(float)
    b = preds["base_prediction"].to_numpy(float)
    h = preds["hybrid_prediction"].to_numpy(float)
    rows = []
    for label, pred in [("F2_base", b), ("F2_hybrid", h)]:
        rows.append({"scope": "pooled_outer_oof", "model": label, **metrics(y, pred)})
    mb, mh = metrics(y, b), metrics(y, h)
    rows.append({"scope": "pooled_outer_oof", "model": "hybrid_minus_base",
                 "delta_pearson": mh["pearson_r"] - mb["pearson_r"], "delta_spearman": mh["spearman_rho"] - mb["spearman_rho"],
                 "delta_r2": mh["r2"] - mb["r2"], "delta_rmse": mh["rmse"] - mb["rmse"], "delta_mae": mh["mae"] - mb["mae"]})
    pd.DataFrame(rows).to_csv(OUT / "base_vs_hybrid_metrics_oof.csv", index=False)
    reg_rows = []
    regimes = [("OCN_lt_40", preds["Opposite_Charge_Number"] < 40), ("OCN_ge_40", preds["Opposite_Charge_Number"] >= 40),
               ("OCN_ge_60", preds["Opposite_Charge_Number"] >= 60)]
    for name, mask in regimes:
        sub = preds[mask]
        if len(sub) < 3:
            continue
        yy = sub["observed"].to_numpy(float)
        bb = sub["base_prediction"].to_numpy(float)
        hh = sub["hybrid_prediction"].to_numpy(float)
        mb, mh = metrics(yy, bb), metrics(yy, hh)
        reg_rows += [{"regime": name, "model": "F2_base", **mb}, {"regime": name, "model": "F2_hybrid", **mh},
                     {"regime": name, "model": "hybrid_minus_base", "n": len(sub),
                      "delta_pearson": mh["pearson_r"] - mb["pearson_r"], "delta_spearman": mh["spearman_rho"] - mb["spearman_rho"],
                      "delta_r2": mh["r2"] - mb["r2"], "delta_rmse": mh["rmse"] - mb["rmse"], "delta_mae": mh["mae"] - mb["mae"]}]
    pd.DataFrame(reg_rows).to_csv(OUT / "base_vs_hybrid_metrics_by_ocn_regime.csv", index=False)


def sensitivity(preds):
    base_m = metrics(preds["observed"], preds["base_prediction"])
    def one(kind, values):
        rows = []
        for v in values:
            center, scale, ub, pct = 40, 12, 0.6, 85
            if kind == "gate_center": center = v
            if kind == "gate_scale": scale = v
            if kind == "lambda_upper": ub = v
            if kind == "cap_percentile": pct = v
            pp = []
            for _, sub in preds.groupby("outer_fold"):
                lam0 = sub["lambda_hat"].iloc[0] if sub["lambda_hat"].iloc[0] <= ub else ub
                cap = float(sub[f"cap_{pct}"].iloc[0])
                raw = np.clip(sub["raw_residual_prediction"].to_numpy(float), -cap, cap)
                pp.extend(sub["base_prediction"].to_numpy(float) + lam0 * gate(sub["Opposite_Charge_Number"].to_numpy(float), center, scale) * raw)
            yy = preds["observed"].to_numpy(float)
            mm = metrics(yy, np.array(pp))
            high = preds["Opposite_Charge_Number"] >= 40
            mh = metrics(yy[high], np.array(pp)[high])
            rows.append({"setting": v, "r2": mm["r2"], "rmse": mm["rmse"], "mae": mm["mae"],
                         "delta_r2_vs_base": mm["r2"] - base_m["r2"], "delta_rmse_vs_base": mm["rmse"] - base_m["rmse"],
                         "high_ocn_r2": mh["r2"], "high_ocn_rmse": mh["rmse"]})
        pd.DataFrame(rows).to_csv(OUT / f"{kind}_sensitivity.csv", index=False)
    one("gate_center", [30, 40, 50])
    one("gate_scale", [8, 12, 16])
    one("lambda_upper", [0.4, 0.6, 0.8])
    one("cap_percentile", [75, 85, 95])


def historical_notes():
    txt = HIST.read_text(errors="replace")
    notes = [
        f"Historical script inspected read-only: {HIST}",
        "1. Historical residual targets appear to be in-sample where base predictions are obtained from a model fitted to the same training data, rather than nested OOF residuals.",
        "2. Residual-fit and calibration observations appear partially overlapping: plateau residuals are fit first and then a tail subset is used for calibration.",
        "3. The historical five-model ensemble propagation is fragile: the script contains paths that pass ensemble predictions/scaler while base_mlp can be None, which can trigger a single-model fallback/guard.",
        "4. Plateau threshold appears to be chosen by quantile logic in the historical implementation rather than as a fully external nested-CV parameter.",
        "5. Historical gate center: 40.",
        "6. Historical gate scale: 12.",
        "7. Historical lambda bound: [0, 0.6].",
        "8. Historical cap definition: percentile cap on absolute calibration residuals, with 85th percentile used in the prior workflow.",
        "9. Historical feature set included the prior larger selected-feature model, not the final F2 parsimonious model.",
        "10. Redundant electrostatic variables in that larger set included OCN/Normalized_OCN and individual charge-related descriptors already audited in Task 1.",
        "",
        "Search evidence:",
    ]
    for token in ["40", "12", "0.6", "percentile", "quantile", "base_mlp", "Normalized_OCN", "Opposite_Charge_Number"]:
        notes.append(f"{token}: {'present' if token in txt else 'not found'}")
    (OUT / "historical_hybrid_implementation_notes.txt").write_text("\n".join(notes) + "\n")


def figures(preds, inner):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(inner["Opposite_Charge_Number"], inner["inner_oof_residual"], s=18, alpha=0.55)
    order = np.argsort(inner["Opposite_Charge_Number"].to_numpy(float))
    xs = inner["Opposite_Charge_Number"].to_numpy(float)[order]
    ys = inner["inner_oof_residual"].to_numpy(float)[order]
    if len(xs) > 20:
        win = max(15, len(xs)//20)
        smooth = pd.Series(ys).rolling(win, center=True, min_periods=5).mean().to_numpy()
        ax.plot(xs, smooth, color="black", lw=2)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(40, color="black", lw=1, ls="--")
    ax.set_xlabel("Opposite_Charge_Number")
    ax.set_ylabel("Inner-OOF F2 residual")
    ax.set_title("Nested training residual structure")
    fig.tight_layout(); fig.savefig(OUT/"residual_vs_ocn.png", dpi=300); fig.savefig(OUT/"residual_vs_ocn.pdf"); plt.close(fig)

    def predplot(path, mask, title):
        sub = preds[mask].copy()
        y = sub["observed"].to_numpy(float)
        lims = [min(y.min(), sub["base_prediction"].min(), sub["hybrid_prediction"].min()), max(y.max(), sub["base_prediction"].max(), sub["hybrid_prediction"].max())]
        pad = 0.05*(lims[1]-lims[0]); lims=[lims[0]-pad, lims[1]+pad]
        fig, axes = plt.subplots(1,2,figsize=(8.4,4.0),sharex=True,sharey=True)
        for ax, col, label in [(axes[0],"base_prediction","F2 base"),(axes[1],"hybrid_prediction","F2 hybrid")]:
            p = sub[col].to_numpy(float); m=metrics(y,p)
            ax.scatter(y,p,s=20,alpha=.65); ax.plot(lims,lims,color="black",lw=1)
            ax.set_title(label); ax.set_xlim(lims); ax.set_ylim(lims)
            ax.text(.04,.96,f"R2={m['r2']:.3f}\nr={m['pearson_r']:.3f}\nRMSE={m['rmse']:.3f}\nn={m['n']}",transform=ax.transAxes,va="top",fontsize=9)
        axes[0].set_ylabel("OOF predicted"); axes[0].set_xlabel("Observed"); axes[1].set_xlabel("Observed")
        fig.suptitle(title); fig.tight_layout(); fig.savefig(OUT/f"{path}.png",dpi=300); fig.savefig(OUT/f"{path}.pdf"); plt.close(fig)
    predplot("base_vs_hybrid_oof", np.ones(len(preds), dtype=bool), "Pooled outer-OOF predictions")
    predplot("base_vs_hybrid_high_ocn", preds["Opposite_Charge_Number"] >= 40, "High-OCN outer-OOF predictions")

    fig, ax = plt.subplots(figsize=(6.2,4.2))
    ax.scatter(preds["Opposite_Charge_Number"], preds["applied_correction"], s=22, alpha=.65)
    ax.axhline(0,color="black",lw=1); ax.axvline(40,color="black",lw=1,ls="--")
    ax.set_xlabel("Opposite_Charge_Number"); ax.set_ylabel("Hybrid - base prediction")
    ax.set_title("Applied residual correction")
    fig.tight_layout(); fig.savefig(OUT/"correction_vs_ocn.png",dpi=300); fig.savefig(OUT/"correction_vs_ocn.pdf"); plt.close(fig)


def write_text(preds, folds, bias, before):
    oof = pd.read_csv(OUT/"base_vs_hybrid_metrics_oof.csv")
    regimes = pd.read_csv(OUT/"base_vs_hybrid_metrics_by_ocn_regime.csv")
    params = pd.read_csv(OUT/"outer_fold_parameters.csv")
    after = {str(p.relative_to(ROOT)): sha256(p) for p in allowed_existing_files()}
    modified = sorted(k for k, v in before.items() if after.get(k) != v)
    missing = sorted(k for k in before if k not in after)
    extra = sorted(k for k in after if k not in before)
    delta = oof[oof["model"]=="hybrid_minus_base"].iloc[0]
    high_delta = regimes[(regimes["regime"]=="OCN_ge_40") & (regimes["model"]=="hybrid_minus_base")].iloc[0]
    low_delta = regimes[(regimes["regime"]=="OCN_lt_40") & (regimes["model"]=="hybrid_minus_base")].iloc[0]
    lines = [
        "F2 hybrid residual ablation, training-only nested OOF",
        "",
        f"Only input table used: {INFILE}",
        f"Rows used: {len(preds)}",
        "The sealed 1,794-pair held-out matrix and unsimulated landscape were not opened, read, hashed, or used.",
        "",
        f"High-OCN inner-OOF residual mean by fold ranged from {bias['mean_residual_high_ocn'].min():.4f} to {bias['mean_residual_high_ocn'].max():.4f}; low-OCN residual mean ranged from {bias['mean_residual_low_ocn'].min():.4f} to {bias['mean_residual_low_ocn'].max():.4f}.",
        f"Unconstrained lambdas by fold: {', '.join(f'{v:.4f}' for v in params['unconstrained_lambda'])}.",
        f"Constrained lambda_hat by fold: {', '.join(f'{v:.4f}' for v in params['lambda_hat'])}.",
        f"Pooled outer-OOF hybrid minus base: delta R2 {delta['delta_r2']:.4f}, delta Pearson {delta['delta_pearson']:.4f}, delta RMSE {delta['delta_rmse']:.4f}, delta MAE {delta['delta_mae']:.4f}.",
        f"OCN >= 40 hybrid minus base: delta R2 {high_delta['delta_r2']:.4f}, delta RMSE {high_delta['delta_rmse']:.4f}, delta MAE {high_delta['delta_mae']:.4f}.",
        f"OCN < 40 hybrid minus base: delta R2 {low_delta['delta_r2']:.4f}, delta RMSE {low_delta['delta_rmse']:.4f}, delta MAE {low_delta['delta_mae']:.4f}.",
        f"Fold-wise delta R2 values: {', '.join(f'{v:.4f}' for v in folds['delta_r2'])}.",
        "",
        "Sensitivity tables report gate-center, gate-scale, lambda-bound, and cap-percentile variants. These were treated as robustness checks and were not used to choose a best setting.",
        "",
        "Conservative interpretation:",
        "Retain the residual correction only if the nested OOF results show consistent high-OCN improvement without degrading the rest of the panel. If lambda_hat collapses toward zero, pooled gains are near zero, or fold-wise improvements are inconsistent, the historical correction is not justified after simplifying the base model to F2 and should be removed from the revised Figure 5 predictor.",
    ]
    (OUT/"analysis_summary.txt").write_text("\n".join(lines)+"\n")
    ver = [
        "Verification",
        f"Only the 917-pair simulation table was used: {INFILE}",
        "The 1,794-pair held-out file was never opened/read/hashed by this script.",
        "Unsimulated predictions were not used.",
        "Outer test folds were isolated from scaling, base training, residual fitting, threshold use, lambda calibration, and cap selection.",
        f"All observations received exactly one outer-OOF base prediction and one outer-OOF hybrid prediction: {len(preds)==917 and preds['row_id'].nunique()==917}",
        f"Residual-fit and calibration subsets were disjoint: {pd.read_csv(OUT/'residual_training_splits.csv').duplicated(['outer_fold','row_id']).sum()==0}",
        f"Pre-existing Figure 5 files modified: {modified}",
        f"Pre-existing Figure 5 files missing: {missing}",
        f"Unexpected new files outside output directory: {extra}",
    ]
    (OUT/"verification.txt").write_text("\n".join(ver)+"\n")
    (OUT/"input_provenance.txt").write_text("\n".join([
        "Input provenance for F2 hybrid residual ablation.",
        f"Input CSV: {INFILE}",
        f"Input SHA256: {sha256(INFILE)}",
        f"Rows expected: 917; rows used: {len(preds)}",
        "Feature set: Opposite_Charge_Number, mean_lambda_RGG, mean_lambda_TAD, SHD_RGG, SHD_TAD.",
        "mean_lambda_RGG=lambda_RGG/RGG_length; mean_lambda_TAD=lambda_TAD/TAD_length.",
        "No held-out matrix or unsimulated prediction landscape was used.",
    ])+"\n")


if __name__ == "__main__":
    main()
