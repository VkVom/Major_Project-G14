from dataclasses import dataclass
from typing import Dict


@dataclass
class FeatureVector:
    sequence_id: int
    frame_id: int
    timestamp: float
    features: Dict[str, float]

    # Exact feature order expected by our trained LSTM
    FEATURE_ORDER = [
        "torso_tilt_deg",
        "hip_velocity_mps",
        "shoulder_accel_mps2",
        "left_knee_angle",
        "right_knee_angle",
        "cog_shift_x"
    ]

    def to_array(self) -> list:
        """
        Convert features into the exact order required by the LSTM.
        """
        return [
            float(self.features[name])
            for name in self.FEATURE_ORDER
        ]


@dataclass
class PredictionResult:
    sequence_id: int
    frame_id: int
    timestamp: float
    state: str
    confidence: float
    probabilities: Dict[str, float]