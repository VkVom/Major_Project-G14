from schemas import FeatureVector
from temporal_pipeline import TemporalPipeline


TEST_CASES = {
    "NORMAL": {
        "torso_tilt_deg": 5.0,
        "hip_velocity_mps": 0.2,
        "shoulder_accel_mps2": 0.1,
        "left_knee_angle": 175.0,
        "right_knee_angle": 174.0,
        "cog_shift_x": 0.02
    },

    "LOSS_OF_BALANCE": {
        "torso_tilt_deg": 20.0,
        "hip_velocity_mps": 0.8,
        "shoulder_accel_mps2": 0.6,
        "left_knee_angle": 155.0,
        "right_knee_angle": 153.0,
        "cog_shift_x": 0.25
    },

    "PRE_FALL": {
        "torso_tilt_deg": 45.0,
        "hip_velocity_mps": 1.6,
        "shoulder_accel_mps2": 1.5,
        "left_knee_angle": 130.0,
        "right_knee_angle": 128.0,
        "cog_shift_x": 0.55
    },

    "FALL": {
        "torso_tilt_deg": 75.0,
        "hip_velocity_mps": 2.8,
        "shoulder_accel_mps2": 3.0,
        "left_knee_angle": 90.0,
        "right_knee_angle": 92.0,
        "cog_shift_x": 0.85
    }
}


for expected_state, features in TEST_CASES.items():

    # Fresh buffer for each test
    pipeline = TemporalPipeline(
        model_path="../models/temporal_model.keras",
        window_size=30
    )

    result = None

    # Create 30 frames
    for frame in range(1, 31):

        feature = FeatureVector(
            sequence_id=1,
            frame_id=frame,
            timestamp=frame / 30.0,
            features=features
        )

        result = pipeline.process(feature)

    print("\n==============================")
    print("Expected :", expected_state)
    print("Predicted:", result.state)
    print("Confidence:", round(result.confidence, 4))

    if result.state == expected_state:
        print("TEST: PASS")
    else:
        print("TEST: FAIL")