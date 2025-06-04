import numpy as np
from collections import defaultdict, deque
import cv2

class HomographyAccumulator:
    """
    Accumulates keypoints across multiple frames to improve homography estimation
    as camera angle changes over time.
    """
    
    def __init__(self, 
                 max_frames=10,           # Maximum frames to keep in memory
                 min_keypoints=4,         # Minimum keypoints needed for homography
                 stability_threshold=5.0, # Pixel distance to consider keypoint "stable"
                 confidence_decay=0.95):  # How much to decay older frame confidence
        
        self.max_frames = max_frames
        self.min_keypoints = min_keypoints
        self.stability_threshold = stability_threshold
        self.confidence_decay = confidence_decay
        
        # Store keypoints for each frame: {frame_num: [(x, y, label, confidence), ...]}
        self.frame_keypoints = {}
        
        # Track which frames we've processed (FIFO queue)
        self.frame_queue = deque(maxlen=max_frames)
        
        # Accumulated stable keypoints: {label: [(x, y, weight, frame_num), ...]}
        self.stable_keypoints = defaultdict(list)
        
        self.current_homography = np.eye(3, dtype=np.float32)
        self.last_update_frame = -1
    
    def add_frame_keypoints(self, frame_num, keypoints, labels, confidences=None):
        """
        Add keypoints from a new frame.
        
        Args:
            frame_num: Frame number
            keypoints: List of (x, y) tuples
            labels: List of keypoint labels (same length as keypoints)
            confidences: Optional list of confidence scores
        """
        if confidences is None:
            confidences = [1.0] * len(keypoints)
        
        # Store this frame's keypoints
        frame_data = [
            (x, y, label, conf) 
            for (x, y), label, conf in zip(keypoints, labels, confidences)
        ]
        self.frame_keypoints[frame_num] = frame_data
        
        # Add to queue and remove old frames if needed
        self.frame_queue.append(frame_num)
        if len(self.frame_queue) > self.max_frames:
            old_frame = self.frame_queue[0]  # Will be removed automatically by deque
            if old_frame in self.frame_keypoints:
                del self.frame_keypoints[old_frame]
        
        # Update stable keypoints
        self._update_stable_keypoints()
    
    def _update_stable_keypoints(self):
        """
        Update the stable keypoints by analyzing consistency across frames.
        """
        self.stable_keypoints.clear()
        
        # Group keypoints by label across all frames
        label_positions = defaultdict(list)
        
        for frame_num in self.frame_queue:
            if frame_num not in self.frame_keypoints:
                continue
                
            for x, y, label, conf in self.frame_keypoints[frame_num]:
                label_positions[label].append((x, y, conf, frame_num))
        
        # For each label, find stable clusters of points
        for label, positions in label_positions.items():
            if len(positions) < 2:  # Need at least 2 observations
                continue
            
            stable_clusters = self._find_stable_clusters(positions)
            
            # Weight clusters by recency and consistency
            for cluster in stable_clusters:
                weighted_points = []
                for x, y, conf, frame_num in cluster:
                    # Apply exponential decay based on frame age
                    age = max(self.frame_queue) - frame_num
                    time_weight = (self.confidence_decay ** age)
                    final_weight = conf * time_weight
                    weighted_points.append((x, y, final_weight, frame_num))
                
                self.stable_keypoints[label].extend(weighted_points)
    
    def _find_stable_clusters(self, positions):
        """
        Group nearby positions into stable clusters.
        Returns list of clusters, where each cluster is a list of (x, y, conf, frame_num).
        """
        if not positions:
            return []
        
        clusters = []
        used = set()
        
        for i, (x1, y1, conf1, frame1) in enumerate(positions):
            if i in used:
                continue
            
            # Start new cluster
            cluster = [(x1, y1, conf1, frame1)]
            used.add(i)
            
            # Find nearby points
            for j, (x2, y2, conf2, frame2) in enumerate(positions):
                if j in used:
                    continue
                
                # Check if point is close to any point in current cluster
                is_close = any(
                    np.sqrt((x2 - cx)**2 + (y2 - cy)**2) < self.stability_threshold
                    for cx, cy, _, _ in cluster
                )
                
                if is_close:
                    cluster.append((x2, y2, conf2, frame2))
                    used.add(j)
            
            # Only keep clusters with multiple observations
            if len(cluster) >= 2:
                clusters.append(cluster)
        
        return clusters
    
    def get_best_keypoints(self):
        """
        Get the best representative keypoints for homography computation.
        Returns (src_pts, dst_pts, labels) suitable for compute_homography.
        """
        best_keypoints = {}
        
        for label, point_list in self.stable_keypoints.items():
            if not point_list:
                continue
            
            # Compute weighted average position
            total_weight = sum(weight for _, _, weight, _ in point_list)
            if total_weight == 0:
                continue
            
            avg_x = sum(x * weight for x, _, weight, _ in point_list) / total_weight
            avg_y = sum(y * weight for _, y, weight, _ in point_list) / total_weight
            
            best_keypoints[label] = (avg_x, avg_y, total_weight)
        
        # Sort by total weight (most stable first) and return top points
        sorted_points = sorted(
            best_keypoints.items(), 
            key=lambda item: item[1][2], 
            reverse=True
        )
        
        return sorted_points
    
    def should_update_homography(self, frame_num, force_update=False):
        """
        Determine if homography should be updated based on accumulated data.
        """
        if force_update:
            return True
        
        # Check if we have enough stable keypoints
        stable_count = len([
            label for label, points in self.stable_keypoints.items() 
            if len(points) >= 2
        ])
        
        if stable_count < self.min_keypoints:
            return False
        
        # Update if it's been a while or we have significantly more data
        frames_since_update = frame_num - self.last_update_frame
        return frames_since_update >= 5 or stable_count > self.min_keypoints * 1.5
    
    def compute_accumulated_homography(self, map_keypoints, frame_num):
        """
        Compute homography using accumulated stable keypoints.
        
        Args:
            map_keypoints: Dictionary mapping label -> (x, y) in map coordinates
            frame_num: Current frame number
            
        Returns:
            Updated homography matrix or None if insufficient data
        """
        best_points = self.get_best_keypoints()
        
        # Filter points that exist in map_keypoints
        valid_points = [
            (label, (x, y), weight) 
            for label, (x, y, weight) in best_points 
            if label in map_keypoints
        ]
        
        if len(valid_points) < self.min_keypoints:
            print(f"Frame {frame_num}: Only {len(valid_points)} valid points, need {self.min_keypoints}")
            return None
        
        # Extract source and destination points
        src_pts = [(x, y) for _, (x, y), _ in valid_points[:20]]  # Limit to top 20
        dst_pts = [map_keypoints[label] for label, _, _ in valid_points[:20]]
        labels = [label for label, _, _ in valid_points[:20]]
        
        print(f"Frame {frame_num}: Computing homography with {len(src_pts)} accumulated points")
        print(f"Labels: {labels}")
        
        try:
            # Import compute_homography from your existing module
            from homography.areas_keypoints_homography import compute_homography
            H = compute_homography(src_pts, dst_pts)
            
            if H is not None:
                self.current_homography = H
                self.last_update_frame = frame_num
                print(f"Frame {frame_num}: Successfully updated homography")
                return H
        except Exception as e:
            print(f"Frame {frame_num}: Homography computation failed: {e}")
        
        return None
    
    def get_current_homography(self):
        """Get the current best homography matrix."""
        return self.current_homography
    
    def get_debug_info(self):
        """Get debug information about accumulated keypoints."""
        info = {
            "frames_in_memory": len(self.frame_queue),
            "stable_keypoint_labels": list(self.stable_keypoints.keys()),
            "keypoint_counts": {
                label: len(points) 
                for label, points in self.stable_keypoints.items()
            }
        }
        return info