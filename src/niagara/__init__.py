"""Niagara: causal discovery from interval-based event sequences."""

from .api import fit
from .config import NiagaraConfig
from .result import CausalMechanism, Endpoint, NiagaraResult

__all__ = [
    "CausalMechanism",
    "Endpoint",
    "NiagaraConfig",
    "NiagaraResult",
    "fit",
]

__version__ = "0.2.0"
