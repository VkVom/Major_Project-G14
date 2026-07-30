from dataclasses import dataclass
from typing import Dict


@dataclass
class FeatureVector:
    sequence_id: int
    frame_id: int
    timestamp: float
    features: Dict[str, float]

    def to_array(self) -> list:
        """Returns ordered float vector for ML model input."""
        return list(self.features.values())


@dataclass
class PredictionResult:
    sequence_id: int
    frame_id: int
    timestamp: float
    state: str
    confidence: float
    probabilities: Dict[str, float]