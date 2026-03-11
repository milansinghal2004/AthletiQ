import json
import os
from angle_calculation import compute_joint_angles


def generate_angle_dataset(pose_file, output_file):

    if not os.path.exists(pose_file):
        print("Pose file not found.")
        return

    with open(pose_file, "r") as f:
        pose_data = json.load(f)

    angle_data = []

    for frame_index, frame_data in enumerate(pose_data):

        angles = compute_joint_angles(frame_data)

        frame_angles = {
            "frame": frame_index,
            **angles
        }

        angle_data.append(frame_angles)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(angle_data, f, indent=2)

    print(f"\nAngle dataset saved to {output_file}")


def main():

    # Read pose data from top-level AthletiQ/outputs/pose_data.json
    # (__file__ is .../src/biomechanics/gen_angle_data.py → go up 3 levels to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    pose_file = os.path.join(project_root, "outputs", "pose_data.json")

    # Save angle data inside top-level AthletiQ/outputs
    output_file = os.path.join(project_root, "outputs", "angle_data.json")

    generate_angle_dataset(pose_file, output_file)


if __name__ == "__main__":
    main()