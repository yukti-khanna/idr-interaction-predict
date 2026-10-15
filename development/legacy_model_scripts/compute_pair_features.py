#!/usr/bin/env python3
"""Compute comprehensive sequence-derived pair features for IDR interaction analysis.

This script is intended to be the main sequence-feature entry point for the
idr-interaction-analysis workflow.

Inputs
------
- canonical pairs CSV with columns: pair_id, idr_a, idr_b
- FASTA file with all IDR sequences
- residues.csv with at least:
    one, lambdas
  and optionally:
    sigmas

Optional input
--------------
- B22 CSV with columns including:
    pair_id, idr_a, idr_b, b22_mean

Output
------
- one CSV row per pair containing sequence features, and optionally B22 columns
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PAIR_COLUMNS = {"pair_id", "idr_a", "idr_b"}
REQUIRED_B22_COLUMNS = {"pair_id", "idr_a", "idr_b", "b22_mean"}

AA_CHARGES = {
    "D": -1,
    "E": -1,
    "K": 1,
    "R": 1,
    "H": 0,
}

HYDROPHOBIC_AA = set("AILMFWVYC")
AROMATIC_AA = set("FWY")
POLAR_AA = set("STNQH")
GLY_AA = {"G"}
PRO_AA = {"P"}
RG_AA = {"R", "G"}
ACIDIC_AA = {"D", "E"}
BASIC_AA = {"R", "K", "H"}


class FeatureComputationError(Exception):
    """Raised when feature calculation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute comprehensive sequence-derived features for canonical IDR pairs."
    )
    parser.add_argument(
        "--pairs-csv",
        required=True,
        type=Path,
        help="Canonical pair table CSV.",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="FASTA file containing all IDR sequences.",
    )
    parser.add_argument(
        "--residues-file",
        required=True,
        type=Path,
        help="Residues CSV containing at least 'one' and 'lambdas', and optionally 'sigmas'.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Output CSV path for pair features.",
    )
    parser.add_argument(
        "--b22-csv",
        type=Path,
        default=None,
        help="Optional B22 summary CSV to merge into the feature table.",
    )
    return parser.parse_args()


def ensure_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FeatureComputationError(f"{label} not found: {path}")
    if not path.is_file():
        raise FeatureComputationError(f"{label} is not a file: {path}")


def read_pairs_table(path: Path) -> list[dict[str, str]]:
    ensure_file_exists(path, "Pairs CSV")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise FeatureComputationError(f"Pairs CSV has no header: {path}")

        missing = REQUIRED_PAIR_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise FeatureComputationError(
                f"Pairs CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            clean = {key: (value or "").strip() for key, value in row.items()}
            if not clean["pair_id"] or not clean["idr_a"] or not clean["idr_b"]:
                raise FeatureComputationError(
                    f"Incomplete row in pairs CSV at line {row_number}: {row}"
                )
            rows.append(
                {
                    "pair_id": clean["pair_id"],
                    "idr_a": clean["idr_a"],
                    "idr_b": clean["idr_b"],
                }
            )

    if not rows:
        raise FeatureComputationError(f"No rows found in pairs CSV: {path}")

    return rows


def read_fasta_sequences(fasta_path: Path) -> dict[str, str]:
    ensure_file_exists(fasta_path, "FASTA file")
    sequences: dict[str, list[str]] = {}
    current_id: str | None = None

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                identifier = line[1:].strip().split()[0]
                if not identifier:
                    raise FeatureComputationError(
                        f"Empty FASTA header encountered at line {line_number} in {fasta_path}."
                    )
                if identifier in sequences:
                    raise FeatureComputationError(
                        f"Duplicate FASTA identifier '{identifier}' found in {fasta_path}."
                    )
                sequences[identifier] = []
                current_id = identifier
            else:
                if current_id is None:
                    raise FeatureComputationError(
                        f"Sequence encountered before first FASTA header at line {line_number}."
                    )
                sequences[current_id].append(line)

    final_sequences = {identifier: "".join(parts) for identifier, parts in sequences.items()}
    if not final_sequences:
        raise FeatureComputationError(f"No sequences found in FASTA file: {fasta_path}")
    return final_sequences


def read_residue_table(path: Path) -> pd.DataFrame:
    ensure_file_exists(path, "Residues file")
    table = pd.read_csv(path)
    required_columns = {"one", "lambdas"}
    missing = required_columns.difference(table.columns)
    if missing:
        raise FeatureComputationError(
            f"Residues file is missing required columns: {', '.join(sorted(missing))}"
        )
    return table.set_index("one")


def read_b22_table(path: Path) -> pd.DataFrame:
    ensure_file_exists(path, "B22 CSV")
    table = pd.read_csv(path)
    missing = REQUIRED_B22_COLUMNS.difference(table.columns)
    if missing:
        raise FeatureComputationError(
            f"B22 CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    return table


def sequence_charge_array(sequence: str) -> np.ndarray:
    return np.array([AA_CHARGES.get(aa, 0) for aa in sequence], dtype=float)


def lambda_array(sequence: str, residue_table: pd.DataFrame) -> np.ndarray:
    missing = [aa for aa in sequence if aa not in residue_table.index]
    if missing:
        raise FeatureComputationError(
            f"Residue(s) not found in residues table: {sorted(set(missing))}"
        )
    return residue_table.loc[list(sequence), "lambdas"].astype(float).to_numpy()


def sigma_array(sequence: str, residue_table: pd.DataFrame) -> np.ndarray | None:
    if "sigmas" not in residue_table.columns:
        return None
    missing = [aa for aa in sequence if aa not in residue_table.index]
    if missing:
        raise FeatureComputationError(
            f"Residue(s) not found in residues table: {sorted(set(missing))}"
        )
    return residue_table.loc[list(sequence), "sigmas"].astype(float).to_numpy()


def safe_mean(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def safe_sum(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(arr.sum())


def safe_std(values: np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.std(ddof=0))


def count_set(sequence: str, aa_set: set[str]) -> int:
    return sum(1 for aa in sequence if aa in aa_set)


def fraction_set(sequence: str, aa_set: set[str]) -> float:
    if not sequence:
        raise FeatureComputationError("Encountered empty sequence while computing fractions.")
    return count_set(sequence, aa_set) / len(sequence)


def count_positive(sequence: str) -> int:
    return sum(1 for aa in sequence if AA_CHARGES.get(aa, 0) > 0)


def count_negative(sequence: str) -> int:
    return sum(1 for aa in sequence if AA_CHARGES.get(aa, 0) < 0)


def count_neutral(sequence: str) -> int:
    return sum(1 for aa in sequence if AA_CHARGES.get(aa, 0) == 0)


def net_charge(sequence: str) -> int:
    return int(sequence_charge_array(sequence).sum())


def ncpr(sequence: str) -> float:
    if not sequence:
        raise FeatureComputationError("Encountered empty sequence while computing NCPR.")
    return float(net_charge(sequence) / len(sequence))


def mean_lambda(sequence: str, residue_table: pd.DataFrame) -> float:
    return safe_mean(lambda_array(sequence, residue_table))


def sum_lambda(sequence: str, residue_table: pd.DataFrame) -> float:
    return safe_sum(lambda_array(sequence, residue_table))


def mean_sigma(sequence: str, residue_table: pd.DataFrame) -> float:
    values = sigma_array(sequence, residue_table)
    return float("nan") if values is None else safe_mean(values)


def sum_sigma(sequence: str, residue_table: pd.DataFrame) -> float:
    values = sigma_array(sequence, residue_table)
    return float("nan") if values is None else safe_sum(values)


def opposite_charge_number(seq_a: str, seq_b: str) -> int:
    pos_a = count_positive(seq_a)
    neg_a = count_negative(seq_a)
    pos_b = count_positive(seq_b)
    neg_b = count_negative(seq_b)
    return int(pos_a * neg_b + neg_a * pos_b)


def normalized_ocn(seq_a: str, seq_b: str) -> float:
    denom = len(seq_a) * len(seq_b)
    if denom == 0:
        raise FeatureComputationError("Encountered empty sequence while computing normalized OCN.")
    return float(opposite_charge_number(seq_a, seq_b) / denom)


def charge_complementarity(seq_a: str, seq_b: str) -> float:
    return float(-(net_charge(seq_a) * net_charge(seq_b)))


def delta_q(seq_a: str, seq_b: str) -> float:
    q_a = net_charge(seq_a)
    q_b = net_charge(seq_b)
    return float(abs(q_a + q_b) - 0.5 * (abs(q_a) + abs(q_b)))


def mean_pair_charge(seq_a: str, seq_b: str) -> float:
    return float((ncpr(seq_a) + ncpr(seq_b)) / 2.0)


def seq_separation_weighted_sum(values: np.ndarray) -> float:
    n = values.size
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += values[i] * values[j] * math.sqrt(j - i)
    return float(total / n)


def compute_scd(sequence: str) -> float:
    charges = sequence_charge_array(sequence)
    return seq_separation_weighted_sum(charges)


def compute_shd(sequence: str, residue_table: pd.DataFrame) -> float:
    lambdas = lambda_array(sequence, residue_table)
    n = lambdas.size
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += ((lambdas[i] + lambdas[j]) / 2.0) / math.sqrt(j - i)
    return float(total / n)


def longest_run_fraction(sequence: str, aa_set: set[str]) -> float:
    if not sequence:
        raise FeatureComputationError("Encountered empty sequence while computing run fractions.")
    longest = 0
    current = 0
    for aa in sequence:
        if aa in aa_set:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest / len(sequence)


def sequence_features(prefix: str, sequence: str, residue_table: pd.DataFrame) -> dict[str, Any]:
    length = len(sequence)
    if length == 0:
        raise FeatureComputationError(f"Sequence '{prefix}' is empty.")

    q = net_charge(sequence)
    seq_ncpr = ncpr(sequence)
    seq_lambda = lambda_array(sequence, residue_table)
    seq_sigma = sigma_array(sequence, residue_table)

    features: dict[str, Any] = {
        prefix: sequence,
        f"{prefix.upper()}_length": length,
        f"charge_{prefix.upper()}": q,
        f"lambda_{prefix.upper()}": safe_mean(seq_lambda),
        f"NCPR_{prefix.upper()}": seq_ncpr,
        f"SCD_{prefix.upper()}": compute_scd(sequence),
        f"SHD_{prefix.upper()}": compute_shd(sequence, residue_table),
    }
    return features


def pair_features(seq_a: str, seq_b: str, residue_table: pd.DataFrame) -> dict[str, Any]:
    len_a = len(seq_a)
    len_b = len(seq_b)
    q_a = net_charge(seq_a)
    q_b = net_charge(seq_b)
    ncpr_a = ncpr(seq_a)
    ncpr_b = ncpr(seq_b)
    lam_a = mean_lambda(seq_a, residue_table)
    lam_b = mean_lambda(seq_b, residue_table)
    sigma_a = mean_sigma(seq_a, residue_table)
    sigma_b = mean_sigma(seq_b, residue_table)
    scd_a = compute_scd(seq_a)
    scd_b = compute_scd(seq_b)
    shd_a = compute_shd(seq_a, residue_table)
    shd_b = compute_shd(seq_b, residue_table)
    ocn = opposite_charge_number(seq_a, seq_b)

    len_sum = len_a + len_b
    len_mean = (len_a + len_b) / 2.0
    len_diff = abs(len_a - len_b)

    min_lambda = min(lam_a, lam_b)
    avg_sigma = float((sigma_a + sigma_b) / 2.0) if np.isfinite(sigma_a) and np.isfinite(sigma_b) else float("nan")
    avg_charge = (q_a + q_b) / 2.0
    avg_ncpr = (ncpr_a + ncpr_b) / 2.0
    avg_scd = (scd_a + scd_b) / 2.0
    avg_shd = (shd_a + shd_b) / 2.0
    avg_interaction = (lam_a + lam_b) / 2.0
    hydropathy_diff = abs(lam_a - lam_b)
    lambda_ratio = float(lam_a / lam_b) if lam_b != 0 else float("nan")
    charge_asymmetry = abs(q_a - q_b)
    total_length = len_sum
    aromatic_a = fraction_set(seq_a, AROMATIC_AA)
    aromatic_b = fraction_set(seq_b, AROMATIC_AA)
    avg_aromatics = (aromatic_a + aromatic_b) / 2.0

    return {
        "Average_lambda": (lam_a + lam_b) / 2.0,
        "Min_lambda": min_lambda,
        "Average_sigma": avg_sigma,
        "Average_charge": avg_charge,
        "Opposite_Charge_Number": ocn,
        "Average_NCPR": avg_ncpr,
        "Charge_Complementarity": charge_complementarity(seq_a, seq_b),
        "Total_Length": total_length,
        "Average_Interaction": avg_interaction,
        "Average_SCD": avg_scd,
        "Average_Aromatics": avg_aromatics,
        "Average_SHD": avg_shd,
        "Normalized_OCN": normalized_ocn(seq_a, seq_b),
        "Charge_Asymmetry": charge_asymmetry,
        "Hydropathy_Diff": hydropathy_diff,
        "Lambda_Ratio": lambda_ratio,
    }


def attach_optional_b22(df: pd.DataFrame, b22_path: Path | None) -> pd.DataFrame:
    if b22_path is None:
        return df

    b22 = read_b22_table(b22_path)
    merged = df.merge(
        b22,
        on=["pair_id", "idr_a", "idr_b"],
        how="left",
        validate="one_to_one",
    )

    if "b22_mean" in merged.columns:
        b22_values = pd.to_numeric(merged["b22_mean"], errors="coerce")
        merged["B22"] = b22_values
        merged["B22_corr1_log10"] = np.where(
            b22_values < 0,
            np.log10(-b22_values),
            np.nan,
        )
        merged["B22_corr2_log10"] = merged["B22_corr1_log10"]
        merged["B22_corr2_log10_no_pos"] = merged["B22_corr1_log10"]
        merged["Corrected_B22_Hummer"] = b22_values
        merged["Corrected_B22_Ganguly"] = b22_values

    return merged


def build_feature_row(
    pair_row: dict[str, str],
    sequences: dict[str, str],
    residue_table: pd.DataFrame,
) -> dict[str, Any]:
    idr_a = pair_row["idr_a"]
    idr_b = pair_row["idr_b"]

    if idr_a not in sequences:
        raise FeatureComputationError(f"Sequence for '{idr_a}' not found in FASTA.")
    if idr_b not in sequences:
        raise FeatureComputationError(f"Sequence for '{idr_b}' not found in FASTA.")

    seq_a = sequences[idr_a]
    seq_b = sequences[idr_b]

    row: dict[str, Any] = {
        "name": pair_row["pair_id"],
        "pair_id": pair_row["pair_id"],
        "idr_a": idr_a,
        "idr_b": idr_b,
    }
    seq_features_a = sequence_features("a", seq_a, residue_table)
    seq_features_b = sequence_features("b", seq_b, residue_table)
    row["a"] = seq_features_a.pop("a")
    row["b"] = seq_features_b.pop("b")
    row.update(seq_features_a)
    row.update(seq_features_b)
    row.update(pair_features(seq_a, seq_b, residue_table))
    return row


def main() -> int:
    args = parse_args()

    try:
        pair_rows = read_pairs_table(args.pairs_csv)
        sequences = read_fasta_sequences(args.fasta)
        residue_table = read_residue_table(args.residues_file)

        output_rows = [
            build_feature_row(
                pair_row=pair_row,
                sequences=sequences,
                residue_table=residue_table,
            )
            for pair_row in pair_rows
        ]
        feature_df = pd.DataFrame(output_rows)
        feature_df = attach_optional_b22(feature_df, args.b22_csv)

        preferred_columns = [
            "name",
            "a",
            "A_length",
            "charge_A",
            "lambda_A",
            "NCPR_A",
            "SCD_A",
            "SHD_A",
            "b",
            "B_length",
            "charge_B",
            "lambda_B",
            "NCPR_B",
            "SCD_B",
            "SHD_B",
            "Average_lambda",
            "Min_lambda",
            "Average_sigma",
            "Average_charge",
            "Opposite_Charge_Number",
            "Average_NCPR",
            "Charge_Complementarity",
            "Total_Length",
            "Average_Interaction",
            "Average_SCD",
            "Average_Aromatics",
            "Average_SHD",
            "Normalized_OCN",
            "Charge_Asymmetry",
            "Hydropathy_Diff",
            "Lambda_Ratio",
            "B22_corr1_log10",
            "B22_corr2_log10",
            "B22_corr2_log10_no_pos",
            "B22",
            "Corrected_B22_Hummer",
            "Corrected_B22_Ganguly",
            "pair_id",
            "idr_a",
            "idr_b",
        ]
        existing_preferred = [col for col in preferred_columns if col in feature_df.columns]
        remaining_columns = [col for col in feature_df.columns if col not in existing_preferred]
        feature_df = feature_df[existing_preferred + remaining_columns]

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(args.output_csv, index=False)

        print(f"Wrote pair features for {len(feature_df)} pair(s) to {args.output_csv}")
        return 0
    except FeatureComputationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ERROR: File system error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())