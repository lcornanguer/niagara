"""Structured public results returned by Niagara."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _plain(value: Any) -> Any:
    """Convert NumPy scalar values to JSON-compatible Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class Endpoint:
    """One start, interval, or end endpoint of an event variable."""

    event: Any
    kind: str


@dataclass(frozen=True)
class CausalMechanism:
    """A selected causal mechanism in the detailed S/I/E graph."""

    parents: Tuple[Endpoint, ...]
    child: Endpoint
    operator: Optional[str] = None


@dataclass(frozen=True)
class NiagaraResult:
    """Result of fitting Niagara to interval-event observations."""

    adjacency: np.ndarray
    feature_names: Tuple[Any, ...]
    mechanisms: Tuple[CausalMechanism, ...]
    horizon: Any

    def __post_init__(self) -> None:
        matrix = np.asarray(self.adjacency)
        expected = (len(self.feature_names), len(self.feature_names))
        if matrix.shape != expected:
            raise ValueError(
                f"adjacency has shape {matrix.shape}; expected {expected}"
            )
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "adjacency", matrix)

    @property
    def edges(self) -> Tuple[Tuple[Any, Any], ...]:
        """Directed event-level edges as ``(parent, child)`` pairs."""
        rows, columns = np.nonzero(self.adjacency)
        return tuple(
            (self.feature_names[row], self.feature_names[column])
            for row, column in zip(rows, columns)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "feature_names": [_plain(name) for name in self.feature_names],
            "adjacency": self.adjacency.tolist(),
            "edges": [[_plain(parent), _plain(child)] for parent, child in self.edges],
            "mechanisms": [
                {
                    "parents": [
                        {"event": _plain(parent.event), "kind": parent.kind}
                        for parent in mechanism.parents
                    ],
                    "child": {
                        "event": _plain(mechanism.child.event),
                        "kind": mechanism.child.kind,
                    },
                    "operator": mechanism.operator,
                }
                for mechanism in self.mechanisms
            ],
            "horizon": (
                [
                    {"sequence": _plain(sequence), "horizon": value}
                    for sequence, value in self.horizon.items()
                ]
                if isinstance(self.horizon, dict)
                else _plain(self.horizon)
            ),
        }

    def to_json(self, path: Any) -> None:
        """Write the structured result to a UTF-8 JSON file."""
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
