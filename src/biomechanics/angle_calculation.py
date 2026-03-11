import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) between three points A-B-C.
    B is the vertex.

    Parameters:
        a, b, c : tuple (x, y)

    Returns:
        angle in degrees
    """

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Prevent division by zero
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Clamp value to avoid floating point errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def get_point(frame_data, joint):
    """
    Extract (x,y) coordinates from frame_data.
    Returns None if joint not found.
    """

    if joint not in frame_data:
        return None

    return (
        frame_data[joint]["x"],
        frame_data[joint]["y"]
    )


def compute_joint_angles(frame_data):
    """
    Compute biomechanical joint angles from a single frame.

    Parameters:
        frame_data : dictionary containing pose landmarks

    Returns:
        dictionary of computed joint angles
    """

    angles = {}

    # Fetch joint coordinates
    left_shoulder = get_point(frame_data, "left_shoulder")
    right_shoulder = get_point(frame_data, "right_shoulder")

    left_elbow = get_point(frame_data, "left_elbow")
    right_elbow = get_point(frame_data, "right_elbow")

    left_wrist = get_point(frame_data, "left_wrist")
    right_wrist = get_point(frame_data, "right_wrist")

    left_hip = get_point(frame_data, "left_hip")
    right_hip = get_point(frame_data, "right_hip")

    left_knee = get_point(frame_data, "left_knee")
    right_knee = get_point(frame_data, "right_knee")

    left_ankle = get_point(frame_data, "left_ankle")
    right_ankle = get_point(frame_data, "right_ankle")

    # --- Elbow Angles ---
    if left_shoulder and left_elbow and left_wrist:
        angles["left_elbow_angle"] = calculate_angle(
            left_shoulder,
            left_elbow,
            left_wrist
        )

    if right_shoulder and right_elbow and right_wrist:
        angles["right_elbow_angle"] = calculate_angle(
            right_shoulder,
            right_elbow,
            right_wrist
        )

    # --- Knee Angles ---
    if left_hip and left_knee and left_ankle:
        angles["left_knee_angle"] = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

    if right_hip and right_knee and right_ankle:
        angles["right_knee_angle"] = calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )

    # --- Hip Angles ---
    if left_shoulder and left_hip and left_knee:
        angles["left_hip_angle"] = calculate_angle(
            left_shoulder,
            left_hip,
            left_knee
        )

    if right_shoulder and right_hip and right_knee:
        angles["right_hip_angle"] = calculate_angle(
            right_shoulder,
            right_hip,
            right_knee
        )

    return angles