"""Command-line interface for Niagara."""

import argparse
import csv
from pathlib import Path
from typing import Any, List

import numpy as np

from . import __version__
from .api import fit
from .config import NiagaraConfig


def _read_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"event", "start", "end"}
        fieldnames = set(reader.fieldnames or ())
        missing = required.difference(fieldnames)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"CSV is missing required column(s): {names}")
        has_sequence = "sequence" in fieldnames
        rows: List[List[Any]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                start = float(row["start"])
                end = None if row["end"].strip() == "" else float(row["end"])
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(
                    f"CSV line {line_number} has an invalid start or end value"
                ) from error
            values: List[Any] = [row["event"], start, end]
            if has_sequence:
                values.append(row["sequence"])
            rows.append(values)
    return np.asarray(rows, dtype=object)


def _write_adjacency(path: Path, names: Any, adjacency: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event"] + list(names))
        for name, row in zip(names, adjacency):
            writer.writerow([name] + [int(value) for value in row])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="niagara",
        description="Causal discovery from interval-based event sequences.",
    )
    parser.add_argument("input", type=Path, help="CSV with event,start,end columns")
    parser.add_argument(
        "--horizon",
        type=float,
        help="Common observation horizon; defaults to each sequence's last timestamp.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Result JSON path")
    parser.add_argument("--adjacency-output", type=Path, help="Optional adjacency CSV path")
    parser.add_argument(
        "--search", choices=("exhaustive", "partial"), default="exhaustive"
    )
    parser.add_argument("--max-error", type=float, default=0.1)
    parser.add_argument("--window-length", type=float, default=30)
    parser.add_argument(
        "--include-end-events",
        action="store_true",
        help="Use the publication implementation's start_only=False mode.",
    )
    parser.add_argument("--no-or-heuristic", action="store_true")
    parser.add_argument("--precision", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Any = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    try:
        data = _read_csv(namespace.input)
        config = NiagaraConfig(
            search=namespace.search,
            max_error=namespace.max_error,
            start_only=not namespace.include_end_events,
            or_heuristic=not namespace.no_or_heuristic,
            precision=namespace.precision,
            window_length=namespace.window_length,
            verbose=namespace.verbose,
        )
        result = fit(data, horizon=namespace.horizon, config=config)
        namespace.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_json(namespace.output)
        adjacency_path = namespace.adjacency_output
        if adjacency_path is not None:
            adjacency_path.parent.mkdir(parents=True, exist_ok=True)
            _write_adjacency(adjacency_path, result.feature_names, result.adjacency)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0
