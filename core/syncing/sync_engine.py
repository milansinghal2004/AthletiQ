import numpy as np

class SyncEngine:
    def __init__(self):
        pass

    def _euclidean_distance(self, v1, v2):
        """Calculate Euclidean distance between two angle vectors."""
        v1 = np.array(v1)
        v2 = np.array(v2)
        # Handle cases where some angles might be missing (empty dicts)
        if v1.size == 0 or v2.size == 0:
            return 1000.0 # High penalty for missing pose
        return np.linalg.norm(v1 - v2)

    def compute_dtw(self, practice_angles, reference_angles):
        """
        Dynamic Time Warping (DTW) to align two sequences.
        practice_angles: list of lists (vectors)
        reference_angles: list of lists (vectors)
        """
        n = len(practice_angles)
        m = len(reference_angles)
        
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

    def sync_videos(self, practice_data, reference_data):
        """
        Sync practice video to reference video metadata.
        Returns a mapping of practice frames to reference frames.
        """
        # Extract angle vectors for each frame
        joints = [
            "left_elbow", "right_elbow",
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee"
        ]
        
        def get_vector(frame):
            angles = frame.get("angles", {})
            if not angles:
                return []
            return [angles.get(j, 0.0) for j in joints]

        practice_vectors = [get_vector(f) for f in practice_data["frames"]]
        reference_vectors = [get_vector(f) for f in reference_data["frames"]]

        # Run DTW
        alignment_path = self.compute_dtw(practice_vectors, reference_vectors)
        
        # Convert path to a frame-by-frame mapping for the practice video
        # We want to know for each practice frame, which reference frame is the closest match
        mapping = {}
        for p_idx, r_idx in alignment_path:
            # If multiple reference frames match a single practice frame, we'll keep the last one or average?
            # Usually, we want a 1-to-1 or many-to-one for playback.
            mapping[p_idx] = r_idx
            
        return mapping
