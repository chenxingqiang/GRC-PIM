"""CiM-ResLDPC: residual-calibrated robust edge RAG on NOR Flash CiM."""

from .config import CiMResLDPCConfig
from .pipeline import CiMResLDPCPipeline, Document, ScoredDoc, RetrieveResult

__all__ = [
    "CiMResLDPCConfig",
    "CiMResLDPCPipeline",
    "Document",
    "ScoredDoc",
    "RetrieveResult",
]

__version__ = "0.1.0"
