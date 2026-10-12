"""Public fitting interface for Niagara."""

from contextlib import nullcontext, redirect_stdout
import io
import math
from numbers import Real
from threading import RLock
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np

from ._algorithm.graph import (
    EndNode,
    Event,
    EventOcc,
    Graph,
    IntervalNode,
    LogicalNode,
    StartNode,
)
from ._algorithm.options import args
from ._algorithm.search import Searcher
from .config import NiagaraConfig
from .result import CausalMechanism, Endpoint, NiagaraResult


class NiagaraCompatibilityWarning(RuntimeWarning):
    """Warning about intentionally preserved publication-code behavior."""


_FIT_LOCK = RLock()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _as_time(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return result


def _rows_from_dataframe(data: Any) -> List[Tuple[Any, Any, Any, Any]]:
    columns = list(data.columns)
    required = {"event", "start", "end"}
    missing = required.difference(columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"DataFrame is missing required column(s): {names}")
    selected = ["event", "start", "end"]
    has_sequence = "sequence" in columns
    if has_sequence:
        selected.append("sequence")
    values = data.loc[:, selected].to_numpy(dtype=object)
    if has_sequence:
        return [tuple(row) for row in values]
    return [(row[0], row[1], row[2], 0) for row in values]


def _rows_from_array(data: Any) -> List[Tuple[Any, Any, Any, Any]]:
    values = np.asarray(data, dtype=object)
    if values.ndim != 2 or values.shape[1] not in {3, 4}:
        raise ValueError(
            "NumPy input must have shape (n_occurrences, 3) for "
            "event/start/end or (n_occurrences, 4) with a sequence column."
        )
    if values.shape[1] == 3:
        return [(row[0], row[1], row[2], 0) for row in values]
    return [tuple(row) for row in values]


def _rows_from_mapping(data: Mapping[Any, Any]) -> List[Tuple[Any, Any, Any, Any]]:
    rows: List[Tuple[Any, Any, Any, Any]] = []
    for event, occurrences in data.items():
        values = np.asarray(occurrences, dtype=object)
        if values.ndim != 2 or values.shape[1] not in {2, 3}:
            raise ValueError(
                f"Occurrences for {event!r} must have shape (n, 2) for "
                "start/end or (n, 3) with a sequence column."
            )
        for row in values:
            sequence = row[2] if values.shape[1] == 3 else 0
            rows.append((event, row[0], row[1], sequence))
    return rows


def _normalise_input(
    data: Any,
    feature_names: Optional[Sequence[Hashable]],
    horizon: float,
    config: NiagaraConfig,
) -> Tuple[Tuple[Hashable, ...], Dict[Hashable, List[EventOcc]], Graph]:
    if isinstance(data, Mapping):
        rows = _rows_from_mapping(data)
    elif hasattr(data, "columns") and hasattr(data, "loc"):
        rows = _rows_from_dataframe(data)
    else:
        rows = _rows_from_array(data)

    if not rows:
        raise ValueError("data must contain at least one interval occurrence")

    if feature_names is None:
        ordered_names: List[Hashable] = []
        for row in rows:
            event = row[0]
            try:
                unseen = event not in ordered_names
            except TypeError as error:
                raise TypeError("event names must be hashable") from error
            if unseen:
                ordered_names.append(event)
    else:
        ordered_names = list(feature_names)
        if not ordered_names:
            raise ValueError("feature_names must not be empty")
        if len(set(ordered_names)) != len(ordered_names):
            raise ValueError("feature_names must be unique")

    events = {name: Event(name) for name in ordered_names}
    occurrences: Dict[Hashable, List[EventOcc]] = {
        name: [] for name in ordered_names
    }

    for index, (raw_event, raw_start, raw_end, sequence) in enumerate(rows):
        event = raw_event
        if event not in events and feature_names is not None:
            if isinstance(event, (int, np.integer)) and 0 <= int(event) < len(ordered_names):
                event = ordered_names[int(event)]
        if event not in events:
            raise ValueError(
                f"Row {index} uses undeclared event {raw_event!r}; add it to feature_names."
            )
        try:
            hash(sequence)
        except TypeError as error:
            raise TypeError(f"sequence at row {index} must be hashable") from error
        start = _as_time(raw_start, f"start at row {index}")
        end = None if _is_missing(raw_end) else _as_time(raw_end, f"end at row {index}")
        if start < 0:
            raise ValueError(f"start at row {index} must be non-negative")
        if start > horizon:
            raise ValueError(f"start at row {index} exceeds horizon {horizon}")
        if end is not None:
            if end < start:
                raise ValueError(f"end at row {index} precedes its start")
            if end > horizon:
                raise ValueError(f"end at row {index} exceeds horizon {horizon}")
        occurrences[event].append(EventOcc(events[event], start, end, sequence))

    empty = [name for name, values in occurrences.items() if not values]
    if empty:
        raise ValueError(f"Every feature must occur at least once; empty: {empty!r}")

    for event, values in occurrences.items():
        starts = [item.start for item in values]
        if starts != sorted(starts):
            warnings.warn(
                f"Occurrences for {event!r} are not ordered by start time. "
                "Publication behavior is row-order-sensitive; the input was not reordered.",
                NiagaraCompatibilityWarning,
                stacklevel=3,
            )

    graph = Graph()
    graph.init_from_events(ordered_names)
    return tuple(ordered_names), occurrences, graph


def _kind(node: Any) -> str:
    if isinstance(node, StartNode):
        return "start"
    if isinstance(node, IntervalNode):
        return "interval"
    if isinstance(node, EndNode):
        return "end"
    raise TypeError(f"Unsupported endpoint node: {type(node).__name__}")


def _mechanisms(graph: Graph) -> Tuple[CausalMechanism, ...]:
    result: List[CausalMechanism] = []
    for child in graph.get_non_logical_nodes():
        if not isinstance(child, (StartNode, EndNode)) or not child.parents:
            continue
        parent = child.parents[0]
        if isinstance(parent, LogicalNode):
            parent_nodes = tuple(parent.parents)
            operator = parent.operator
        else:
            parent_nodes = (parent,)
            operator = None
        if all(node.event == child.event for node in parent_nodes):
            continue
        result.append(
            CausalMechanism(
                parents=tuple(Endpoint(node.event.name, _kind(node)) for node in parent_nodes),
                child=Endpoint(child.event.name, _kind(child)),
                operator=operator,
            )
        )
    return tuple(result)


def fit(
    data: Any,
    *,
    horizon: Real,
    feature_names: Optional[Sequence[Hashable]] = None,
    config: Optional[NiagaraConfig] = None,
) -> NiagaraResult:
    """Fit Niagara to interval-based event observations.

    ``data`` may be a Pandas DataFrame with ``event``, ``start``, ``end``,
    and optional ``sequence`` columns; a NumPy array with the same three or
    four columns; or a mapping from event names to start/end arrays.
    """
    effective = config if config is not None else NiagaraConfig()
    numeric_horizon = _as_time(horizon, "horizon")
    if numeric_horizon <= 0:
        raise ValueError("horizon must be positive")
    if effective.search == "partial" and effective.window_length != 30:
        warnings.warn(
            "The publication implementation ignores window_length and uses 30; "
            "the supplied value is retained in the configuration but not applied.",
            NiagaraCompatibilityWarning,
            stacklevel=2,
        )
    if not effective.start_only:
        warnings.warn(
            "start_only=False uses the unchanged end-event path from the "
            "publication snapshot, including its known limitations.",
            NiagaraCompatibilityWarning,
            stacklevel=2,
        )

    names, occurrences, empty_graph = _normalise_input(
        data, feature_names, numeric_horizon, effective
    )

    with _FIT_LOCK:
        previous = {
            "n": args.n,
            "search": args.search,
            "window_length": args.window_length,
            "err_max": args.err_max,
            "precision": args.precision,
            "duration_distrib": args.duration_distrib,
            "or_heuristic": args.or_heuristic,
            "start_only": args.start_only,
        }
        args.n = numeric_horizon
        args.search = effective.search
        args.window_length = effective.window_length
        args.err_max = effective.max_error
        args.precision = effective.precision
        args.duration_distrib = effective.duration_distribution
        args.or_heuristic = effective.or_heuristic
        args.start_only = effective.start_only
        output = nullcontext() if effective.verbose else redirect_stdout(io.StringIO())
        try:
            with output:
                searcher = Searcher(
                    numeric_horizon,
                    occurrences,
                    empty_graph,
                    exhaustive_search=effective.search == "exhaustive",
                    verbose=effective.verbose,
                )
                searcher.main()
        finally:
            for name, value in previous.items():
                setattr(args, name, value)

    adjacency = searcher.graph.as_simple_dag_adj_matrix()
    return NiagaraResult(
        adjacency=adjacency,
        feature_names=names,
        mechanisms=_mechanisms(searcher.graph),
        horizon=numeric_horizon,
    )
