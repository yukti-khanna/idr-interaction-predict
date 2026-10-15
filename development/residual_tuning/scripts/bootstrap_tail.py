#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def vals(y, p):
    resid = y - p
    return {
        "delta_RMSE": np.sqrt(np.mean((y - p) ** 2)),
        "delta_MAE": np.mean(np.abs(y - p)),
        "delta_bias": np.mean(resid),
        "delta_Pearson": np.corrcoef(y, p)[0, 1],
        "delta_Spearman": stats.spearmanr(y, p).statistic,
        "delta_R2": 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2),
    }


def main():
    p = argparse.ArgumentParser(description="Paired bootstrap on unique high-OCN averaged OOF predictions.")
    p.add_argument("--predictions", default="outputs/outer_oof_predictions.csv")
    p.add_argument("--output", default="outputs")
    p.add_argument("--threshold", type=float, default=40.0)
    p.add_argument("--n-resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260912)
    args = p.parse_args()
    out = Path(args.output)
    pred = pd.read_csv(args.predictions)
    rows = []
    rng = np.random.default_rng(args.seed)
    for label_value, group in pred.groupby("residual_model_label"):
        unique = group.groupby("name").agg(
            true_target=("true_target", "first"),
            OCN=("OCN", "first"),
            base_prediction=("base_prediction", "mean"),
            hybrid_prediction=("hybrid_prediction", "mean"),
            n_repeats=("outer_seed", "nunique"),
        ).reset_index()
        high = unique[unique["OCN"] >= args.threshold].reset_index(drop=True)
        y = high["true_target"].to_numpy(float)
        b = high["base_prediction"].to_numpy(float)
        h = high["hybrid_prediction"].to_numpy(float)
        obs_b = vals(y, b)
        obs_h = vals(y, h)
        boot = {k: [] for k in obs_b}
        for _ in range(args.n_resamples):
            idx = rng.integers(0, len(high), len(high))
            vb = vals(y[idx], b[idx])
            vh = vals(y[idx], h[idx])
            for k in boot:
                boot[k].append(vh[k] - vb[k])
        for k, arr in boot.items():
            arr = np.asarray(arr)
            rows.append({
                "comparison": f"{label_value}_hybrid_minus_base",
                "residual_model_label": label_value,
                "metric": k,
                "observed_difference": obs_h[k] - obs_b[k],
                "bootstrap_mean": float(np.mean(arr)),
                "ci_2_5": float(np.percentile(arr, 2.5)),
                "ci_97_5": float(np.percentile(arr, 97.5)),
                "n_unique_high_OCN_pairs": len(high),
                "n_resamples": args.n_resamples,
            })
    pd.DataFrame(rows).to_csv(out / "tail_bootstrap_differences.csv", index=False)


if __name__ == "__main__":
    main()
