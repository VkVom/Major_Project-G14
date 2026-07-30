from schemas import FeatureVector


sample_feature = FeatureVector(
    sequence_id=1,
    frame_id=10,
    timestamp=123.45,
    features={
        "torso_tilt_deg": 5.2,
        "hip_velocity_mps": 0.1,
        "left_knee_angle": 175.0
    }
)

print(sample_feature)
print(sample_feature.to_array())