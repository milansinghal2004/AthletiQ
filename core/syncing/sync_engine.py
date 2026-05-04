import numpy as np

class SyncEngine:
    def __init__(self):
        pass

    def _euclidean_distance(self, v1, v2):
        """Calculate Euclidean distance between two angle vectors."""
        v1 = np.array(v1)
        v2 = np.array(v2)
        # Handle cases where some angles might be missing (empty lists)
        if v1.size == 0 or v2.size == 0:
            return 1000.0 # High penalty for missing pose
        return np.linalg.norm(v1 - v2)

    def compute_dtw(self, practice_angles, reference_angles):
        """
        Dynamic Time Warping (DTW) to align two sequences.
        """
        n = len(practice_angles)
        m = len(reference_angles)
        
        if n == 0 or m == 0:
            return []
            
        # Initialize cost matrix
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._euclidean_distance(practice_angles[i-1], reference_angles[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                             dtw_matrix[i, j-1],    # deletion
                                             dtw_matrix[i-1, j-1]) # match
        
        # Backtrack to find the optimal path
        path = []
        i, j = n, m
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            options = [dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]]
            best = np.argmin(options)
            if best == 0:
                i -= 1
            elif best == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        
        path.reverse()
        return path

    def identify_phases(self, pose_data):
        """
        Identify key phases (start, strike, end) in the pose data.
        The 'strike' is estimated as the moment where the wrists are at their lowest point.
        """
        frames = pose_data.get("frames", [])
        if not frames:
            return None

        wrist_y_values = []
        for f in frames:
            landmarks = f.get("landmarks", {})
            # MediaPipe Y: 0 at top, 1 at bottom.
            # We want the max Y (lowest point in image) for the 'strike'.
            ly = landmarks.get("left_wrist", {}).get("y")
            ry = landmarks.get("right_wrist", {}).get("y")
            
            if ly is not None and ry is not None:
                wrist_y_values.append((ly + ry) / 2)
            elif ly is not None:
                wrist_y_values.append(ly)
            elif ry is not None:
                wrist_y_values.append(ry)
            else:
                # If no wrist landmarks, use a default or previous value
                wrist_y_values.append(wrist_y_values[-1] if wrist_y_values else 0.5)

        if not wrist_y_values:
            return None

        # Convert to numpy for easier analysis
        wrist_y = np.array(wrist_y_values)
        
        # Smooth the values to avoid noise-driven peaks
        window = min(5, len(wrist_y))
        if window > 1:
            wrist_y = np.convolve(wrist_y, np.ones(window)/window, mode='same')

        strike_idx = np.argmax(wrist_y)
        
        return {
            "start": 0,
            "strike": int(strike_idx),
            "end": len(frames) - 1
        }

    def sync_videos(self, practice_data, reference_data):
        """
        Sync practice video to reference video using Segmented DTW.
        Returns the full alignment path (list of frame pairs) and phase data.
        """
        # 1. Identify phases for both videos
        p_phases = self.identify_phases(practice_data)
        r_phases = self.identify_phases(reference_data)

        if not p_phases or not r_phases:
            print("Warning: Phases could not be identified. Falling back to global DTW.")
            return self._sync_global(practice_data, reference_data)

        # 2. Extract feature vectors
        joint_weights = {
            "left_elbow": 1.5, "right_elbow": 1.5,
            "left_shoulder": 1.2, "right_shoulder": 1.2,
            "left_hip": 1.0, "right_hip": 1.0,
            "left_knee": 0.8, "right_knee": 0.8
        }
        joints = list(joint_weights.keys())

        def get_vector(frame):
            angles = frame.get("angles", {})
            if not angles: return []
            return [angles.get(j, 0.0) * joint_weights[j] for j in joints]

        p_vectors = [get_vector(f) for f in practice_data["frames"]]
        r_vectors = [get_vector(f) for f in reference_data["frames"]]

        # 3. Perform Segmented DTW
        # Segment A: Start to Strike
        p_seg_a = p_vectors[:p_phases['strike'] + 1]
        r_seg_a = r_vectors[:r_phases['strike'] + 1]
        path_a = self.compute_dtw(p_seg_a, r_seg_a)

        # Segment B: Strike to End
        p_seg_b = p_vectors[p_phases['strike']:]
        r_seg_b = r_vectors[r_phases['strike']:]
        path_b = self.compute_dtw(p_seg_b, r_seg_b)

        # 4. Merge paths
        full_path = []
        for p_idx, r_idx in path_a:
            full_path.append((p_idx, r_idx))
        
        p_offset = p_phases['strike']
        r_offset = r_phases['strike']
        # Skip the first pair of path_b as it's the strike point (already in path_a)
        for p_idx, r_idx in path_b[1:]:
            full_path.append((p_idx + p_offset, r_idx + r_offset))

        return full_path, p_phases, r_phases

    def _sync_global(self, practice_data, reference_data):
        """Original unsegmented DTW for fallback."""
        joint_weights = {
            "left_elbow": 1.5, "right_elbow": 1.5,
            "left_shoulder": 1.2, "right_shoulder": 1.2,
            "left_hip": 1.0, "right_hip": 1.0,
            "left_knee": 0.8, "right_knee": 0.8
        }
        joints = list(joint_weights.keys())

        def get_vector(frame):
            angles = frame.get("angles", {})
            if not angles: return []
            return [angles.get(j, 0.0) * joint_weights[j] for j in joints]

        p_vectors = [get_vector(f) for f in practice_data["frames"]]
        r_vectors = [get_vector(f) for f in reference_data["frames"]]
        
        alignment_path = self.compute_dtw(p_vectors, r_vectors)
        return alignment_path, None, None
