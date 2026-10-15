#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Make basic high-OCN residual tuning figures.")
    p.add_argument("--output", default="outputs")
    p.add_argument("--threshold", type=float, default=40.0)
    args = p.parse_args()
    out = Path(args.output)
    pred = pd.read_csv(out / "outer_oof_predictions.csv")
    for label_value, group in pred.groupby("residual_model_label"):
        high = group[group["OCN"] >= args.threshold].copy()
        high["base_residual"] = high["true_target"] - high["base_prediction"]
        high["hybrid_residual"] = high["true_target"] - high["hybrid_prediction"]
        safe = str(label_value).replace("/", "_")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(high["OCN"], high["base_residual"], s=15, alpha=0.45, label="Base")
        ax.scatter(high["OCN"], high["hybrid_residual"], s=15, alpha=0.45, label=f"{label_value} hybrid")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Opposite_Charge_Number")
        ax.set_ylabel("Residual")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"high_OCN_residual_vs_OCN_{safe}.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4))
        data = [(high["true_target"] - high["base_prediction"]).abs(), (high["true_target"] - high["hybrid_prediction"]).abs()]
        ax.boxplot(data, labels=["Base", f"{label_value} hybrid"], showfliers=False)
        ax.set_ylabel("Absolute error")
        fig.tight_layout()
        fig.savefig(out / f"high_OCN_absolute_error_{safe}.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
