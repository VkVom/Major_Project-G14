import numpy as np

from model import TemporalModelInference

# Load model
model = TemporalModelInference("../models/temporal_model.keras")

# Create a sample sequence
sequence = np.random.rand(30, 6).astype(np.float32)

# Predict
result = model.predict(
    sequence_matrix=sequence,
    sequence_id=1,
    frame_id=30,
    timestamp=0.99
)

print(result)