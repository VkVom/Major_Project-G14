from schemas import FeatureVector
from temporal_pipeline import TemporalPipeline


# Load our complete M3 pipeline
pipeline = TemporalPipeline(
    model_path="../models/temporal_model.keras",
    window_size=30
)


# Send 31 sample frames
for frame in range(1, 32):

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

    result = pipeline.process(feature)

    if result is None:
        print(f"Frame {frame}: Waiting for 30 frames...")

    else:
        print(f"\nFrame {frame}: PREDICTION")
        print("State:", result.state)
        print("Confidence:", round(result.confidence, 4))
        print("Probabilities:", result.probabilities)