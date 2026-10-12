"""Configuration for Niagara."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NiagaraConfig:
    """Scientific configuration for a Niagara fit.

    Defaults match the publication implementation. ``window_length`` is
    retained for compatibility, although publication code always uses 30.
    """

    search: str = "exhaustive"
    max_error: float = 0.1
    start_only: bool = True
    or_heuristic: bool = True
    precision: int = 2
    window_length: float = 30
    duration_distribution: str = "LogNormal"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.search not in {"exhaustive", "partial"}:
            raise ValueError("search must be 'exhaustive' or 'partial'")
        if not 0 <= self.max_error <= 1:
            raise ValueError("max_error must be between 0 and 1")
        if self.precision < 0:
            raise ValueError("precision must be non-negative")
        if self.window_length <= 0:
            raise ValueError("window_length must be positive")
        if self.duration_distribution != "LogNormal":
            raise ValueError(
                "The publication implementation supports only "
                "duration_distribution='LogNormal'."
            )
