from collections import deque
import numpy as np

from schemas import FeatureVector


class TemporalBuffer:
    """
    Stores the latest FeatureVector frames for temporal prediction.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def append(self, feature: FeatureVector) -> None:
        """Add one FeatureVector to the buffer."""
        self.buffer.append(feature)

    def is_full(self) -> bool:
        """Check whether 30 frames are available."""
        return len(self.buffer) == self.window_size

    def size(self) -> int:
        """Return current number of frames."""
        return len(self.buffer)

    def get_sequence(self) -> np.ndarray:
        """
        Convert buffered FeatureVectors into the
        (30, 6) matrix required by the LSTM.
        """

        if not self.is_full():
            raise ValueError(
                f"Buffer is not full. "
                f"Current frames: {len(self.buffer)}/{self.window_size}"
            )

        sequence = [
            feature.to_array()
            for feature in self.buffer
        ]

        return np.array(sequence, dtype=np.float32)

    def clear(self) -> None:
        """Remove all frames."""
        self.buffer.clear()