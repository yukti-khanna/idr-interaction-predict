#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd


F2_P8_FEATURES = [
    "Opposite_Charge_Number",
    "mean_lambda_RGG",
    "mean_lambda_TAD",
    "SHD_RGG",
    "SHD_TAD",
    "opp_charge_std",
    "scd_patch1_std",
    "scd_patch2_std",
    "rg_fraction_patch1_std",
    "aromatic_fraction_patch2_std",
    "shd_patch1_std",
    "shd_patch2_std",
    "ah_energy_std",
]


def main():
    p = argparse.ArgumentParser(description="Recreate the packaged merged 917-row F2-P8 input table.")
    p.add_argument("--base", required=True, help="Authoritative Figure 5 917-pair base CSV.")
    p.add_argument("--patch", required=True, help="Historical patch feature CSV.")
    p.add_argument("--output", default="inputs/fig5_917_F2_P8.csv")
    args = p.parse_args()
    base = pd.read_csv(args.base)
    patch = pd.read_csv(args.patch)
    patch_cols = F2_P8_FEATURES[5:]
    if len(base) != 917 or len(patch) != 917:
        raise SystemExit(f"Expected 917 rows in each input; observed base={len(base)}, patch={len(patch)}")
    if not base["name"].is_unique or not patch["name"].is_unique:
        raise SystemExit("Expected unique name identifiers in both inputs")
    if set(base["name"]) != set(patch["name"]):
        raise SystemExit("Base and patch name sets differ")
    df = base.copy()
    df["mean_lambda_RGG"] = df["lambda_RGG"] / df["RGG_length"]
    df["mean_lambda_TAD"] = df["lambda_TAD"] / df["TAD_length"]
    df = df.merge(patch[["name"] + patch_cols], on="name", how="left", validate="one_to_one", sort=False)
    cols = ["name", "rgg", "tad", "B22_corr1_log10"] + F2_P8_FEATURES
    out = df[cols].copy()
    if out[["B22_corr1_log10"] + F2_P8_FEATURES].isna().any().any():
        raise SystemExit("Missing values found after merge")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(out)} rows")


if __name__ == "__main__":
    main()
