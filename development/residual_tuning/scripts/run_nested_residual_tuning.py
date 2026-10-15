#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    apply_hybrid,
    calibrate,
    calculate_metrics,
    completed_fold_path,
    enumerate_candidates,
    fit_base_predict,
    load_input_table,
    load_json,
    make_inner_oof_base_predictions,
    make_outer_folds,
    predict_residual_model,
    set_all_seeds,
    soft_gate,
    train_residual_model,
)


def parse_args():
    p = argparse.ArgumentParser(description="Strict nested high-OCN residual tuning for Figure 5 F2-P8.")
    p.add_argument("--input", required=True, help="Merged 917-pair F2-P8 input CSV.")
    p.add_argument("--config", required=True, help="Run configuration JSON.")
    p.add_argument("--grid", required=True, help="Residual hyperparameter grid JSON.")
    p.add_argument("--base-config", default="config/base_model.json", help="Base F2-P8 model configuration JSON.")
    p.add_argument("--output", required=True, help="Output directory.")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and enumerate workload, then exit before training.")
    p.add_argument("--resume", action="store_true", help="Skip outer folds with completed checkpoints.")
    return p.parse_args()


def split_high_indices(ocn_train, threshold, seed):
    high = np.where(ocn_train >= threshold)[0]
    fit_idx, cal_idx = train_test_split(high, test_size=0.35, random_state=seed, shuffle=True)
    return np.asarray(fit_idx), np.asarray(cal_idx)


def evaluate_candidate(candidate, X_train_s, y_train, residual_targets, ocn_train, fit_idx, cal_idx, run_config, seed):
    sample_weight = soft_gate(ocn_train[fit_idx], run_config)
    model, train_info = train_residual_model(
        candidate,
        X_train_s[fit_idx],
        residual_targets[fit_idx],
        sample_weight,
        run_config,
        seed,
    )
    pred_cal = predict_residual_model(model, X_train_s[cal_idx])
    lambda_raw, lambda_hat, cap = calibrate(residual_targets[cal_idx], pred_cal, run_config)
    hybrid_cal, _, _ = apply_hybrid(
        np.zeros(len(cal_idx)),
        pred_cal,
        ocn_train[cal_idx],
        lambda_hat,
        cap,
        run_config,
    )
    # Selection evaluates how well calibrated residual predictions explain calibration residuals.
    met = calculate_metrics(residual_targets[cal_idx], hybrid_cal)
    row = {
        **candidate,
        "lambda_raw": lambda_raw,
        "lambda_hat": lambda_hat,
        "cap": cap,
        "inner_tail_RMSE": met["RMSE"],
        "inner_tail_MAE": met["MAE"],
        "inner_tail_R2": met["R2"],
        "inner_tail_Pearson": met["Pearson"],
        "inner_tail_Spearman": met["Spearman"],
        **train_info,
    }
    return row, model


def choose_best(rows, run_config):
    table = pd.DataFrame(rows).copy()
    tol = float(run_config["selection"]["tie_tolerance"])
    best_rmse = table["inner_tail_RMSE"].min()
    tied = table[table["inner_tail_RMSE"] <= best_rmse + tol].copy()
    best_mae = tied["inner_tail_MAE"].min()
    tied = tied[tied["inner_tail_MAE"] <= best_mae + tol].copy()
    tied = tied.sort_values(["complexity", "model_type", "ridge_alpha"], na_position="last")
    selected_idx = tied.index[0]
    table["selected"] = False
    table.loc[selected_idx, "selected"] = True
    return table, table.loc[selected_idx].to_dict()


def select_representative_rows(selection_table):
    reps = []
    tuned = selection_table[selection_table["selected"]].iloc[0]
    reps.append(("tuned", tuned))
    ridge = selection_table[selection_table["model_type"] == "Ridge"].copy()
    if len(ridge):
        ridge = ridge.sort_values(["inner_tail_RMSE", "inner_tail_MAE", "complexity"])
        reps.append(("ridge", ridge.iloc[0]))
    hist = selection_table[
        (selection_table["model_type"] == "NN")
        & (selection_table["architecture"].astype(str) == "(16, 8)")
        & (selection_table["l2"].astype(float) == 0.0005)
        & (selection_table["learning_rate"].astype(float) == 0.0002)
        & (selection_table["huber_delta"].astype(float) == 0.75)
    ]
    if len(hist):
        reps.append(("historical_fixed_nn", hist.iloc[0]))
    return reps


def run_outer_fold(df, base_config, run_config, candidates, outer_seed, outer_fold, train_idx, val_idx, output_dir):
    threshold = float(run_config["high_ocn_threshold"])
    seed = int(outer_seed * 100 + outer_fold)
    determinism_status = set_all_seeds(seed)
    features = base_config["features"]
    target = base_config["target"]
    X = df[features].to_numpy(float)
    y = df[target].to_numpy(float)
    ocn = df["Opposite_Charge_Number"].to_numpy(float)
    names = df["name"].to_numpy()

    print(f"Outer seed {outer_seed} fold {outer_fold}/5", flush=True)
    print(f"High-OCN training n = {int((ocn[train_idx] >= threshold).sum())}; validation n = {int((ocn[val_idx] >= threshold).sum())}", flush=True)
    print(f"Testing {len(candidates)} candidate residual models...", flush=True)

    run_config = dict(run_config)
    run_config["_current_outer_seed"] = outer_seed
    base_val, outer_scaler = fit_base_predict(X[train_idx], y[train_idx], X[val_idx], base_config, run_config)
    inner_oof = make_inner_oof_base_predictions(X[train_idx], y[train_idx], base_config, run_config)
    residual_targets = y[train_idx] - inner_oof
    X_train_s = outer_scaler.transform(X[train_idx])
    X_val_s = outer_scaler.transform(X[val_idx])
    fit_idx, cal_idx = split_high_indices(ocn[train_idx], threshold, seed)

    selection_rows = []
    trained_models = {}
    for i, candidate in enumerate(candidates, start=1):
        cseed = seed * 1000 + i
        row, model = evaluate_candidate(candidate, X_train_s, y[train_idx], residual_targets, ocn[train_idx], fit_idx, cal_idx, run_config, cseed)
        row.update({"outer_seed": outer_seed, "outer_fold": outer_fold})
        selection_rows.append(row)
        trained_models[i - 1] = model

    selection_table, selected = choose_best(selection_rows, run_config)
    print(f"Current best tail RMSE = {selected['inner_tail_RMSE']:.5f}", flush=True)
    print(f"Selected: {selected['model_type']} {selected['architecture']} {selected}", flush=True)

    pred_rows = []
    fold_results = []
    val_high = ocn[val_idx] >= threshold
    base_tail = calculate_metrics(y[val_idx][val_high], base_val[val_high])
    for residual_label, rep in select_representative_rows(selection_table):
        model_pos = int(rep.name)
        chosen_model = trained_models[model_pos]
        pred_cal = predict_residual_model(chosen_model, X_train_s[cal_idx])
        lambda_raw, lambda_hat, cap = calibrate(residual_targets[cal_idx], pred_cal, run_config)
        residual_raw_val = predict_residual_model(chosen_model, X_val_s)
        hybrid_val, residual_clipped, correction = apply_hybrid(base_val, residual_raw_val, ocn[val_idx], lambda_hat, cap, run_config)
        for local, bp, rr, rc, corr, hp in zip(val_idx, base_val, residual_raw_val, residual_clipped, correction, hybrid_val):
            pred_rows.append({
                "name": names[local],
                "outer_seed": outer_seed,
                "outer_fold": outer_fold,
                "residual_model_label": residual_label,
                "true_target": y[local],
                "OCN": ocn[local],
                "base_prediction": bp,
                "residual_prediction_raw": rr,
                "residual_prediction_clipped": rc,
                "gate": soft_gate([ocn[local]], run_config)[0],
                "lambda_raw": lambda_raw,
                "lambda_hat": lambda_hat,
                "cap": cap,
                "applied_correction": corr,
                "hybrid_prediction": hp,
                "chosen_model_type": rep["model_type"],
                "chosen_architecture": str(rep["architecture"]),
            })
        hybrid_tail = calculate_metrics(y[val_idx][val_high], hybrid_val[val_high])
        if residual_label == "tuned":
            print(f"Outer tail RMSE: base={base_tail['RMSE']:.5f}, tuned_hybrid={hybrid_tail['RMSE']:.5f}", flush=True)
        fold_results.append({
            "outer_seed": outer_seed,
            "outer_fold": outer_fold,
            "residual_model_label": residual_label,
            "chosen_model_type": rep["model_type"],
            "chosen_architecture": str(rep["architecture"]),
            "chosen_hyperparameters": json.dumps({
                "ridge_alpha": rep.get("ridge_alpha"),
                "l2": rep.get("l2"),
                "learning_rate": rep.get("learning_rate"),
                "huber_delta": rep.get("huber_delta"),
            }),
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "n_high_train": int((ocn[train_idx] >= threshold).sum()),
            "n_high_validation": int(val_high.sum()),
            "lambda_raw": lambda_raw,
            "lambda_hat": lambda_hat,
            "cap": cap,
            "base_tail_RMSE": base_tail["RMSE"],
            "hybrid_tail_RMSE": hybrid_tail["RMSE"],
            "base_tail_MAE": base_tail["MAE"],
            "hybrid_tail_MAE": hybrid_tail["MAE"],
            "base_tail_bias": base_tail["mean_residual"],
            "hybrid_tail_bias": hybrid_tail["mean_residual"],
            "base_tail_R2": base_tail["R2"],
            "hybrid_tail_R2": hybrid_tail["R2"],
            "base_tail_Pearson": base_tail["Pearson"],
            "hybrid_tail_Pearson": hybrid_tail["Pearson"],
            "base_tail_Spearman": base_tail["Spearman"],
            "hybrid_tail_Spearman": hybrid_tail["Spearman"],
            "determinism_status": determinism_status,
        })

    split_rows = []
    for idx, split in [(fit_idx, "residual_fit"), (cal_idx, "calibration")]:
        for i in idx:
            global_idx = train_idx[i]
            split_rows.append({
                "outer_seed": outer_seed,
                "outer_fold": outer_fold,
                "split": split,
                "name": names[global_idx],
                "OCN": ocn[global_idx],
            })

    return selection_table, pd.DataFrame(pred_rows), pd.DataFrame(fold_results), pd.DataFrame(split_rows)


def append_csv(path, df):
    path = Path(path)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    base_config = load_json(args.base_config)
    run_config = load_json(args.config)
    grid = load_json(args.grid)
    df = load_input_table(args.input, base_config)
    folds = make_outer_folds(len(df), run_config)
    candidates = enumerate_candidates(grid)
    nn_count = sum(c["model_type"] == "NN" for c in candidates)
    ridge_count = sum(c["model_type"] == "Ridge" for c in candidates)
    expected_base_ensemble_fits = len(folds) * (1 + run_config["inner_cv"]["n_splits"]) * len(base_config["mlp"]["ensemble_random_seeds"])

    if args.dry_run:
        print("DRY RUN PASS")
        print(f"Input rows: {len(df)}")
        print(f"Unique pair IDs: {df['name'].nunique()}")
        print(f"Outer folds: {len(folds)}")
        print(f"Residual candidates: total={len(candidates)}, ridge={ridge_count}, nn={nn_count}")
        print(f"Expected base MLP fits: {expected_base_ensemble_fits}")
        print(f"Expected residual model fits: {len(folds) * len(candidates)}")
        print(f"High-OCN pairs in full table: {int((df['Opposite_Charge_Number'] >= run_config['high_ocn_threshold']).sum())}")
        print(f"Output path writable: {output_dir.exists()}")
        print("Exiting before model training.")
        return

    for outer_seed, outer_fold, train_idx, val_idx in folds:
        done = completed_fold_path(output_dir, outer_seed, outer_fold)
        if args.resume and done.exists():
            print(f"Skipping completed outer seed {outer_seed} fold {outer_fold}", flush=True)
            continue
        selection, pred, fold_result, splits = run_outer_fold(
            df, base_config, run_config, candidates, outer_seed, outer_fold, train_idx, val_idx, output_dir
        )
        append_csv(output_dir / "inner_tuning_results.csv", selection)
        append_csv(output_dir / "outer_oof_predictions.csv", pred)
        append_csv(output_dir / "outer_fold_results.csv", fold_result)
        append_csv(output_dir / "residual_fit_calibration_ids.csv", splits)
        done.write_text("complete\n")

    print("Nested residual tuning complete. Run evaluate_tail.py, bootstrap_tail.py, and make_figures.py.", flush=True)


if __name__ == "__main__":
    main()
