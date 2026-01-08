"""Model Serving Module"""

from .predictor import FraudPredictor
from .api import app

__all__ = ["FraudPredictor", "app"]
