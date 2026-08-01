from schemas import FeatureVector
from buffer import TemporalBuffer


buffer = TemporalBuffer(window_size=30)

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

print("Frames:", buffer.size())
print("Buffer full:", buffer.is_full())

sequence = buffer.get_sequence()

print("Sequence shape:", sequence.shape)
print(sequence)