from schemas import FeatureVector
from buffer import TemporalBuffer
from model import TemporalModelInference


# 1. Create 30-frame buffer
buffer = TemporalBuffer(window_size=30)

# 2. Load trained LSTM model
model = TemporalModelInference(
    "../models/temporal_model.keras"
)


# 3. Simulate 30 FeatureVectors coming from Module 2
for frame in range(30):

    feature = FeatureVector(
        sequence_id=1,
        frame_id=frame,
        timestamp=frame / 30.0,
        features={
            "torso_tilt_deg": 5.0,
            "hip_velocity_mps": 0.2,
            "shoulder_accel_mps2": 0.1,
            "left_knee_angle": 175.0,
            "right_knee_angle": 174.0,
            "cog_shift_x": 0.02
        }
    )

    buffer.append(feature)


# 4. Check whether 30 frames are ready
if buffer.is_full():

    sequence = buffer.get_sequence()

    # Get metadata from latest frame
    latest_feature = buffer.buffer[-1]

    # 5. Send sequence to LSTM
    result = model.predict(
        sequence_matrix=sequence,
        sequence_id=latest_feature.sequence_id,
        frame_id=latest_feature.frame_id,
        timestamp=latest_feature.timestamp
    )

    print("\n===== MODULE 3 RESULT =====")
    print("Sequence shape:", sequence.shape)
    print("State:", result.state)
    print("Confidence:", result.confidence)
    print("Probabilities:", result.probabilities)

else:
    print("Waiting for 30 frames...")