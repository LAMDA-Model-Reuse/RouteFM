"""RouteFM: pretrained in-context routing with interchangeable encoders."""

from routefm.api import (
    DEFAULT_BGE_MODEL,
    DEFAULT_BGE_REVISION,
    ContextObservation,
    RouteFMRouter,
    RoutingDecision,
)

__all__ = [
    "ContextObservation",
    "DEFAULT_BGE_MODEL",
    "DEFAULT_BGE_REVISION",
    "RouteFMRouter",
    "RoutingDecision",
    "__version__",
]

__version__ = "1.1.0"
