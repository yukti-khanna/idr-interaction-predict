#!/usr/bin/env python3
"""Predict IDR pair interaction strength from a feature CSV using the calibrated hybrid bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pickle
import io

import joblib
import numpy as np
import pandas as pd


SELECTED_COLS = [
    "Opposite_Charge_Number",
    "Normalized_OCN",
    "charge_TAD",
    "Average_SHD",
    "lambda_RGG",
    "Lambda_Ratio",
    "lambda_TAD",
    "TAD_length",
    "Average_Interaction",
    "SHD_TAD",
]


class PredictionError(Exception):
    pass


def load_bundle_compat(bundle_path: Path):
    """Load older joblib/pickle bundles with NumPy RNG compatibility shims.

    This handles bundles pickled in older environments where MT19937 may be
    stored as the class repr string rather than the canonical bit-generator
    name.
    """
    mt_repr = str(np.random.MT19937)
    mt_name = getattr(np.random.MT19937, "__name__", "MT19937")

    try:
        import numpy.random.bit_generator as _bg
        import numpy.random._pickle as _np_pickle

        if hasattr(_bg, "BitGenerators"):
            _bg.BitGenerators.setdefault(mt_repr, np.random.MT19937)
            _bg.BitGenerators.setdefault(mt_name, np.random.MT19937)
            _bg.BitGenerators.setdefault("MT19937", np.random.MT19937)

        if hasattr(_np_pickle, "BitGenerators"):
            _np_pickle.BitGenerators.setdefault(mt_repr, np.random.MT19937)
            _np_pickle.BitGenerators.setdefault(mt_name, np.random.MT19937)
            _np_pickle.BitGenerators.setdefault("MT19937", np.random.MT19937)

        if hasattr(_np_pickle, "__bit_generator_ctor"):
            _orig_bg_ctor = _np_pickle.__bit_generator_ctor

            def _compat_bg_ctor(bit_generator_name='MT19937'):
                if bit_generator_name in {mt_repr, mt_name, 'MT19937'}:
                    return np.random.MT19937()
                return _orig_bg_ctor(bit_generator_name)

            _np_pickle.__bit_generator_ctor = _compat_bg_ctor

        if hasattr(_np_pickle, "__generator_ctor"):
            _orig_gen_ctor = _np_pickle.__generator_ctor

            def _compat_gen_ctor(bit_generator_name='MT19937', bit_generator_ctor=None):
                if bit_generator_name in {mt_repr, mt_name, 'MT19937'}:
                    return np.random.Generator(np.random.MT19937())
                return _orig_gen_ctor(bit_generator_name, bit_generator_ctor)

            _np_pickle.__generator_ctor = _compat_gen_ctor
    except Exception:
        pass

    raw = bundle_path.read_bytes()
    patched = raw.replace(mt_repr.encode("utf-8"), b"MT19937")

    errors = []
    loaders = (
        lambda p: joblib.load(p),
        lambda p: pickle.load(open(p, "rb")),
        lambda _: joblib.load(io.BytesIO(patched)),
        lambda _: pickle.loads(patched),
    )
    for loader in loaders:
        try:
            return loader(bundle_path)
        except Exception as exc:
            errors.append(exc)

    raise errors[-1]

def normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    alias_map = {
        "a": "rgg",
        "b": "tad",
        "A_length": "RGG_length",
        "B_length": "TAD_length",
        "charge_A": "charge_RGG",
        "charge_B": "charge_TAD",
        "lambda_A": "lambda_RGG",
        "lambda_B": "lambda_TAD",
        "NCPR_A": "NCPR_RGG",
        "NCPR_B": "NCPR_TAD",
        "SCD_A": "SCD_RGG",
        "SCD_B": "SCD_TAD",
        "SHD_A": "SHD_RGG",
        "SHD_B": "SHD_TAD",
    }

    for src, dst in alias_map.items():
        if dst not in df.columns and src in df.columns:
            df = df.rename(columns={src: dst})

    return df

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Normalized_OCN" not in df.columns:
        df["Normalized_OCN"] = df["Opposite_Charge_Number"] / df["Total_Length"]
    if "Charge_Asymmetry" not in df.columns and {"charge_TAD", "charge_RGG"}.issubset(df.columns):
        df["Charge_Asymmetry"] = np.abs(df["charge_TAD"] - df["charge_RGG"])
    if "Hydropathy_Diff" not in df.columns and {"SHD_TAD", "SHD_RGG"}.issubset(df.columns):
        df["Hydropathy_Diff"] = df["SHD_TAD"] - df["SHD_RGG"]
    if "Lambda_Ratio" not in df.columns and {"lambda_TAD", "lambda_RGG"}.issubset(df.columns):
        df["Lambda_Ratio"] = df["lambda_TAD"] / (df["lambda_RGG"] + 1e-6)
    return df


def soft_gate(ocn_vec: np.ndarray, center: float = 40.0, scale: float = 12.0) -> np.ndarray:
    z = (ocn_vec - center) / scale
    return 1.0 / (1.0 + np.exp(-z))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict pairwise interaction strength with pretrained calibrated hybrid model.")
    p.add_argument("--features-csv", required=True, type=Path, help="Input pair feature table.")
    p.add_argument("--bundle", required=True, type=Path, help="joblib bundle with base_mlp, scaler, residual_nn, threshold, lambda_hat, cap.")
    p.add_argument("--output-csv", required=True, type=Path, help="Output CSV with predictions.")
    p.add_argument("--ranked-output-csv", required=True, type=Path, help="Output ranked CSV.")
    p.add_argument("--prediction-column", default="pred_B22_corr1_log10", help="Prediction column name.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not args.features_csv.exists():
            raise PredictionError(f"Features CSV not found: {args.features_csv}")
        if not args.bundle.exists():
            raise PredictionError(f"Bundle not found: {args.bundle}")

        df = pd.read_csv(args.features_csv)
        df = normalize_feature_columns(df)
        df = add_engineered_features(df)

        missing = [c for c in SELECTED_COLS if c not in df.columns]
        if missing:
            raise PredictionError(f"Missing required feature columns: {missing}")

        bundle = load_bundle_compat(args.bundle)
        if not isinstance(bundle, dict):
            raise PredictionError(f"Loaded bundle is not a dict. Got: {type(bundle)}")

        required_keys = {"base_mlp", "scaler", "residual_nn", "threshold", "lambda_hat", "cap"}
        if not required_keys.issubset(bundle.keys()):
            raise PredictionError(
                f"Bundle missing required keys: {sorted(required_keys.difference(bundle.keys()))}. "
                f"Bundle keys are: {sorted(bundle.keys())}"
            )

        base_mlp = bundle["base_mlp"]
        scaler = bundle["scaler"]
        residual_nn = bundle["residual_nn"]
        threshold = float(bundle["threshold"])
        lambda_hat = float(bundle["lambda_hat"])
        cap = float(bundle["cap"])

        X = df[SELECTED_COLS].to_numpy(dtype=float)
        Xs = scaler.transform(X)

        y_base = np.asarray(base_mlp.predict(Xs), dtype=float)
        y_corr = np.asarray(residual_nn.predict(Xs, verbose=0)).ravel().astype(float)
        y_corr = np.clip(y_corr, -cap, cap)

        ocn = df["Opposite_Charge_Number"].to_numpy(dtype=float)
        gate = soft_gate(ocn, center=40.0, scale=12.0)
        y_hybrid = y_base + (lambda_hat * gate * y_corr)

        out = df.copy()
        out["pred_base"] = y_base
        out["pred_residual"] = y_corr
        out["pred_gate"] = gate
        out[args.prediction_column] = y_hybrid

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output_csv, index=False)

        ranked = out.sort_values(args.prediction_column, ascending=False).reset_index(drop=True)
        args.ranked_output_csv.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.ranked_output_csv, index=False)

        print(f"Wrote predictions to {args.output_csv}")
        print(f"Wrote ranked predictions to {args.ranked_output_csv}")
        return 0

    except PredictionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: Unexpected prediction failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
