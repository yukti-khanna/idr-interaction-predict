#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Validate completed residual tuning outputs.")
    p.add_argument("--output", default="outputs")
    args = p.parse_args()
    out = Path(args.output)
    required = [
        "inner_tuning_results.csv",
        "outer_fold_results.csv",
        "outer_oof_predictions.csv",
        "residual_fit_calibration_ids.csv",
        "tail_metrics.csv",
        "high_OCN_pair_improvements.csv",
        "tail_bootstrap_differences.csv",
    ]
    failures = []
    for f in required:
        pth = out / f
        if not pth.exists() or pth.stat().st_size == 0:
            failures.append(f)
    if failures:
        raise SystemExit(f"Missing or empty outputs: {failures}")
    pred = pd.read_csv(out / "outer_oof_predictions.csv")
    if pred[["name", "outer_seed", "outer_fold"]].duplicated().any():
        raise SystemExit("Duplicate outer OOF predictions detected")
    print("Output validation PASS")


if __name__ == "__main__":
    main()
