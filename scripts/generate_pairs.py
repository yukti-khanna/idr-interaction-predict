#!/usr/bin/env python3
"""Generate a canonical IDR pair table from a FASTA file.

This script supports two input modes:
1. Explicit pair list mode via `--pairs-file`, where each line is of the form
   `idr_a__idr_b`.
2. Two-list mode via `--list-a` and `--list-b`, where all pairwise combinations
   are generated between the two sets of IDs.

Pairs are validated against the FASTA headers, canonicalized so that
`idr_a__idr_b` and `idr_b__idr_a` are treated as the same pair, and written as a
CSV table for downstream workflow use.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable


PAIR_SEPARATOR = "__"


class PairGenerationError(Exception):
    """Raised when input validation or pair generation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a canonical CSV table of IDR pairs from a FASTA file and "
            "either an explicit pairs file or two ID lists."
        )
    )
    parser.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="FASTA file containing all IDR sequences.",
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=None,
        help=(
            "Optional text file with one pair per line in the format "
            "'idr_a__idr_b'."
        ),
    )
    parser.add_argument(
        "--list-a",
        type=Path,
        default=None,
        help="Optional text file containing one IDR identifier per line.",
    )
    parser.add_argument(
        "--list-b",
        type=Path,
        default=None,
        help="Optional text file containing one IDR identifier per line.",
    )
    parser.add_argument(
        "--allow-self-pairs",
        action="store_true",
        help="Allow pairs of the form idr_x__idr_x.",
    )
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help=(
            "Keep pair order exactly as provided instead of canonicalizing. "
            "By default, pair members are sorted alphabetically so reversed "
            "pairs collapse to the same output pair."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV path for the generated pair table.",
    )
    return parser.parse_args()


def read_fasta_ids(fasta_path: Path) -> list[str]:
    if not fasta_path.exists():
        raise PairGenerationError(f"FASTA file not found: {fasta_path}")

    ids: list[str] = []
    seen: set[str] = set()

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                identifier = line[1:].strip().split()[0]
                if not identifier:
                    raise PairGenerationError(
                        f"Empty FASTA header encountered at line {line_number} in {fasta_path}."
                    )
                if identifier in seen:
                    raise PairGenerationError(
                        f"Duplicate FASTA identifier '{identifier}' found in {fasta_path}."
                    )
                seen.add(identifier)
                ids.append(identifier)

    if not ids:
        raise PairGenerationError(f"No FASTA headers found in {fasta_path}.")

    return ids


def read_id_list(list_path: Path, label: str) -> list[str]:
    if not list_path.exists():
        raise PairGenerationError(f"{label} file not found: {list_path}")

    values: list[str] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if PAIR_SEPARATOR in line:
                raise PairGenerationError(
                    f"Unexpected pair separator '{PAIR_SEPARATOR}' in {label} file "
                    f"{list_path} at line {line_number}: '{line}'"
                )
            values.append(line)

    if not values:
        raise PairGenerationError(f"No IDs found in {label} file: {list_path}")

    return values


def parse_pair_line(line: str, source: Path, line_number: int) -> tuple[str, str]:
    if line.count(PAIR_SEPARATOR) != 1:
        raise PairGenerationError(
            f"Invalid pair format in {source} at line {line_number}: '{line}'. "
            f"Expected exactly one '{PAIR_SEPARATOR}'."
        )

    left, right = (part.strip() for part in line.split(PAIR_SEPARATOR))
    if not left or not right:
        raise PairGenerationError(
            f"Invalid pair in {source} at line {line_number}: '{line}'. "
            "Pair members cannot be empty."
        )

    return left, right


def iter_pairs_from_file(pairs_file: Path) -> Iterable[tuple[str, str]]:
    if not pairs_file.exists():
        raise PairGenerationError(f"Pairs file not found: {pairs_file}")

    with pairs_file.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield parse_pair_line(line, pairs_file, line_number)


def iter_pairs_from_lists(list_a: list[str], list_b: list[str]) -> Iterable[tuple[str, str]]:
    for left in list_a:
        for right in list_b:
            yield left, right


def canonicalize_pair(idr_a: str, idr_b: str, keep_order: bool) -> tuple[str, str, str]:
    if keep_order:
        left, right = idr_a, idr_b
    else:
        left, right = sorted((idr_a, idr_b))
    pair_id = f"{left}{PAIR_SEPARATOR}{right}"
    return pair_id, left, right


def validate_identifier(identifier: str, valid_ids: set[str], context: str) -> None:
    if identifier not in valid_ids:
        raise PairGenerationError(
            f"Unknown IDR identifier '{identifier}' encountered in {context}. "
            "It is not present in the FASTA file."
        )


def generate_pair_rows(
    *,
    pair_source: Iterable[tuple[str, str]],
    fasta_ids: list[str],
    allow_self_pairs: bool,
    keep_order: bool,
) -> list[dict[str, str]]:
    fasta_id_set = set(fasta_ids)
    unique_pairs: OrderedDict[str, tuple[str, str]] = OrderedDict()

    for idr_a, idr_b in pair_source:
        validate_identifier(idr_a, fasta_id_set, f"pair '{idr_a}{PAIR_SEPARATOR}{idr_b}'")
        validate_identifier(idr_b, fasta_id_set, f"pair '{idr_a}{PAIR_SEPARATOR}{idr_b}'")

        if not allow_self_pairs and idr_a == idr_b:
            continue

        pair_id, left, right = canonicalize_pair(idr_a, idr_b, keep_order=keep_order)
        unique_pairs.setdefault(pair_id, (left, right))

    if not unique_pairs:
        raise PairGenerationError(
            "No valid pairs were generated after validation and duplicate removal."
        )

    rows: list[dict[str, str]] = []
    for pair_id, (idr_a, idr_b) in unique_pairs.items():
        rows.append(
            {
                "pair_id": pair_id,
                "idr_a": idr_a,
                "idr_b": idr_b,
            }
        )
    return rows


def write_pairs_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "idr_a", "idr_b"])
        writer.writeheader()
        writer.writerows(rows)


def validate_mode(args: argparse.Namespace) -> str:
    explicit_mode = args.pairs_file is not None
    list_mode = args.list_a is not None or args.list_b is not None

    if explicit_mode and list_mode:
        raise PairGenerationError(
            "Choose exactly one input mode: either --pairs-file or both --list-a and --list-b."
        )

    if explicit_mode:
        return "pairs_file"

    if args.list_a is not None and args.list_b is not None:
        return "two_lists"

    raise PairGenerationError(
        "No valid input mode provided. Use --pairs-file or both --list-a and --list-b."
    )


def main() -> int:
    args = parse_args()

    try:
        mode = validate_mode(args)
        fasta_ids = read_fasta_ids(args.fasta)

        if mode == "pairs_file":
            pair_source = iter_pairs_from_file(args.pairs_file)
        else:
            list_a = read_id_list(args.list_a, "list_a")
            list_b = read_id_list(args.list_b, "list_b")
            for identifier in list_a:
                validate_identifier(identifier, set(fasta_ids), f"list_a file '{args.list_a}'")
            for identifier in list_b:
                validate_identifier(identifier, set(fasta_ids), f"list_b file '{args.list_b}'")
            pair_source = iter_pairs_from_lists(list_a, list_b)

        rows = generate_pair_rows(
            pair_source=pair_source,
            fasta_ids=fasta_ids,
            allow_self_pairs=args.allow_self_pairs,
            keep_order=args.keep_order,
        )
        write_pairs_csv(rows, args.output)

        print(f"Wrote {len(rows)} canonical pair(s) to {args.output}")
        return 0
    except PairGenerationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
