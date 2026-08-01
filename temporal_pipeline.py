from buffer import TemporalBuffer
from model import TemporalModelInference
from schemas import FeatureVector, PredictionResult


class TemporalPipeline:
    """
    Complete Module 3 pipeline.

    Receives one FeatureVector at a time.
    After 30 frames are available, runs the temporal model
    and returns a PredictionResult.
    """

    def __init__(
        self,
        model_path: str,
        window_size: int = 30
    ):
        self.buffer = TemporalBuffer(window_size=window_size)
        self.model = TemporalModelInference(model_path)

    def process(
        self,
        feature: FeatureVector
    ) -> PredictionResult | None:

        # Add latest M2 feature vector
        self.buffer.append(feature)

        # Not enough frames yet
        if not self.buffer.is_full():
            return None

        # Convert 30 frames into model input
        sequence = self.buffer.get_sequence()

        # Run LSTM prediction
        result = self.model.predict(
            sequence_matrix=sequence,
            sequence_id=feature.sequence_id,
            frame_id=feature.frame_id,
            timestamp=feature.timestamp
        )

        return result