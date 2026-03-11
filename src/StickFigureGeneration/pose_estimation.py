import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import warnings
warnings.filterwarnings("ignore")

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp
import json
import time
from biomechanics.angle_calculation import compute_joint_angles


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Required landmarks for cricket biomechanics
REQUIRED_LANDMARKS = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle"
}


class PoseEstimator:

    def __init__(self):

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        keypoints = {}

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
            )

            h, w, _ = frame.shape

            for idx, lm in enumerate(results.pose_landmarks.landmark):

                if idx in REQUIRED_LANDMARKS:

                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    joint_name = REQUIRED_LANDMARKS[idx]

                    keypoints[joint_name] = {
                        "x": x,
                        "y": y,
                        "visibility": lm.visibility
                    }

                    # Draw joint point
                    cv2.circle(frame, (x, y), 5, (0,255,255), -1)

                    # Optional debug label
                    cv2.putText(
                        frame,
                        joint_name,
                        (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255,255,255),
                        1
                    )

        return frame, keypoints


def run_pose_detection(source, save_data=False):

    angle_data = []

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error opening video source")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    pose_model = PoseEstimator()

    pose_data = []

    frame_count = 0  # FIX 1

    prev_time = 0

    cv2.namedWindow("AthletiQ Pose Detection", cv2.WINDOW_NORMAL)

    print("\nPress 'q' to exit\n")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (960,540))

        frame, keypoints = pose_model.process_frame(frame)

        # Save pose data
        pose_data.append(keypoints)  # FIX 2

        if keypoints:  # FIX 4
            angles = compute_joint_angles(keypoints)

            angle_data.append({
                "frame": frame_count,
                **angles
            })

        frame_count += 1  # FIX 1

        # FPS Calculation
        curr_time = time.time()
        fps_display = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps_display)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("AthletiQ Pose Detection", frame)

        key = cv2.waitKey(delay)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if save_data:

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        outputs_dir = os.path.join(project_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        pose_output_path = os.path.join(outputs_dir, "pose_data.json")
        angle_output_path = os.path.join(outputs_dir, "angle_data.json")

        with open(pose_output_path, "w") as f:
            json.dump(pose_data, f)

        with open(angle_output_path, "w") as f:
            json.dump(angle_data, f, indent=2)

        print(f"\nPose data saved to {pose_output_path}")
        print(f"Angle data saved to {angle_output_path}")

        
def main():

    print("\n===== AthletiQ Pose Detection =====\n")

    print("Select Input Mode")
    print("1 → Webcam")
    print("2 → Video File")

    choice = input("\nEnter choice: ")

    if choice == "1":

        save = input("Save pose data to JSON? (y/n): ")

        run_pose_detection(
            source=0,
            save_data=(save.lower() == "y")
        )

    elif choice == "2":

        video_path = input("Enter video path: ")

        if not os.path.exists(video_path):
            print("Invalid path")
            return

        save = input("Save pose data to JSON? (y/n): ")

        run_pose_detection(
            source=video_path,
            save_data=(save.lower() == "y")
        )

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()