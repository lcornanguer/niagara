"""Mutable compatibility options used by the publication implementation.

The original code parsed command-line arguments at import time and exposed them
as a module global. The package adapter updates this object for one protected
fit call at a time so the numerical implementation can remain unchanged.
"""

from dataclasses import dataclass


@dataclass
class _LegacyOptions:
    n: float = 500000
    search: str = "exhaustive"
    window_length: float = 30
    err_max: float = 0.1
    precision: int = 2
    duration_distrib: str = "LogNormal"
    or_heuristic: bool = True
    start_only: bool = True


args = _LegacyOptions()
