"""Niagara: causal discovery from interval-based event sequences."""

from .api import NiagaraCompatibilityWarning, fit
from .config import NiagaraConfig
from .result import CausalMechanism, Endpoint, NiagaraResult

__all__ = [
    "CausalMechanism",
    "Endpoint",
    "NiagaraCompatibilityWarning",
    "NiagaraConfig",
    "NiagaraResult",
    "fit",
]

__version__ = "0.1.0"
