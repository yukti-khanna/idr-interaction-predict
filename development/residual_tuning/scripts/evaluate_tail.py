#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import calculate_metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate base and tuned hybrid outer-OOF predictions.")
    p.add_argument("--predictions", default="outputs/outer_oof_predictions.csv")
    p.add_argument("--output", default="outputs")
    p.add_argument("--threshold", type=float, default=40.0)
    args = p.parse_args()
    out = Path(args.output)
    pred = pd.read_csv(args.predictions)
    regions = {
        "ALL": np.ones(len(pred), dtype=bool),
        "LOW": pred["OCN"].to_numpy(float) < args.threshold,
        "HIGH": pred["OCN"].to_numpy(float) >= args.threshold,
    }
    rows = []
    for label_value, group in pred.groupby("residual_model_label"):
        for region, mask_all in regions.items():
            mask = mask_all & (pred["residual_model_label"] == label_value).to_numpy()
            for label, col in [("base", "base_prediction"), (f"{label_value}_hybrid", "hybrid_prediction")]:
                row = {"residual_model_label": label_value, "model": label, "region": region}
                row.update(calculate_metrics(pred.loc[mask, "true_target"], pred.loc[mask, col]))
                rows.append(row)
    pd.DataFrame(rows).to_csv(out / "tail_metrics.csv", index=False)
    highs = []
    summaries = []
    for label_value, group in pred.groupby("residual_model_label"):
        high = group[group["OCN"] >= args.threshold].copy()
        high["abs_error_base"] = (high["true_target"] - high["base_prediction"]).abs()
        high["abs_error_hybrid"] = (high["true_target"] - high["hybrid_prediction"]).abs()
        high["delta_abs_error"] = high["abs_error_base"] - high["abs_error_hybrid"]
        highs.append(high)
        summaries.append({
            "residual_model_label": label_value,
            "fraction_improved": float((high["delta_abs_error"] > 0).mean()),
            "fraction_worsened": float((high["delta_abs_error"] < 0).mean()),
            "mean_improvement": float(high["delta_abs_error"].mean()),
            "median_improvement": float(high["delta_abs_error"].median()),
        })
    pd.concat(highs, ignore_index=True).to_csv(out / "high_OCN_pair_improvements.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "high_OCN_pair_improvement_summary.csv", index=False)


if __name__ == "__main__":
    main()
