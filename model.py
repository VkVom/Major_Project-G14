import numpy as np
import tensorflow as tf

from schemas import PredictionResult


class TemporalModelInference:
    """
    Loads the trained Keras model and performs temporal inference.
    """

    CLASSES = [
        "NORMAL",
        "LOSS_OF_BALANCE",
        "PRE_FALL",
        "FALL"
    ]

    def __init__(self, model_path: str):
        """
        Load the trained Keras model only once.
        """
        self.model = tf.keras.models.load_model(model_path)

    def predict(
        self,
        sequence_matrix: np.ndarray,
        sequence_id: int,
        frame_id: int,
        timestamp: float
    ) -> PredictionResult:
        """
        Run inference on one temporal sequence.
        """

        # Add batch dimension
        input_data = np.expand_dims(
            sequence_matrix,
            axis=0
        ).astype(np.float32)

        # Predict probabilities
        probabilities = self.model.predict(
            input_data,
            verbose=0
        )[0]

        # Highest probability class
        top_index = int(np.argmax(probabilities))

        predicted_state = self.CLASSES[top_index]

        confidence = float(probabilities[top_index])

        probability_dict = {
            class_name: float(probabilities[i])
            for i, class_name in enumerate(self.CLASSES)
        }

        return PredictionResult(
            sequence_id=sequence_id,
            frame_id=frame_id,
            timestamp=timestamp,
            state=predicted_state,
            confidence=confidence,
            probabilities=probability_dict
        )