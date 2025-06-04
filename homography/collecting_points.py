import numpy as np
import cv2
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

class CameraAngle(Enum):
    """Different camera perspectives"""
    WIDE_CENTER = "wide_center"
    LEFT_SIDE = "left_side" 
    RIGHT_SIDE = "right_side"
    BEHIND_GOAL_1 = "behind_goal_1"
    BEHIND_GOAL_2 = "behind_goal_2"
    CORNER_1 = "corner_1"
    CORNER_2 = "corner_2"
    UNKNOWN = "unknown"

@dataclass
class PerspectiveObservation:
    """Keypoint observation with camera perspective info"""
    image_pt: Tuple[float, float]  # (x, y) in image coordinates
    field_pt: Tuple[float, float]  # (x, y) in field coordinates
    frame_id: int
    camera_angle: CameraAngle
    confidence: float = 1.0
    detection_quality: float = 1.0  # how clear/visible was this keypoint

class MultiPerspectiveHomographyEstimator:
    def __init__(self, 
                 max_observations_per_angle: int = 200,
                 min_points_per_homography: int = 8,
                 perspective_weight_decay: float = 0.95,  # newer perspectives weighted higher
                 spatial_diversity_bonus: float = 2.0):
        
        self.max_observations_per_angle = max_observations_per_angle
        self.min_points_per_homography = min_points_per_homography
        self.perspective_weight_decay = perspective_weight_decay
        self.spatial_diversity_bonus = spatial_diversity_bonus
        
        # Store observations per camera angle
        self.observations_by_angle: Dict[CameraAngle, deque] = defaultdict(
            lambda: deque(maxlen=max_observations_per_angle)
        )
        
        # Individual homographies per angle (for comparison/validation)
        self.homographies_by_angle: Dict[CameraAngle, np.ndarray] = {}
        
        # Master homography combining all perspectives
        self.master_homography: Optional[np.ndarray] = None
        self.last_update_frame = 0
        
        # Track which field regions are well-covered by each angle
        self.coverage_maps: Dict[CameraAngle, np.ndarray] = {}
        
    def add_perspective_keypoints(self, 
                                image_pts: List[Tuple[float, float]], 
                                field_pts: List[Tuple[float, float]], 
                                frame_id: int,
                                camera_angle: CameraAngle,
                                confidences: Optional[List[float]] = None,
                                detection_qualities: Optional[List[float]] = None) -> bool:
        """Add keypoint observations from a specific camera perspective"""
        
        if len(image_pts) != len(field_pts):
            raise ValueError("Image and field point lists must have same length")
            
        if confidences is None:
            confidences = [1.0] * len(image_pts)
        if detection_qualities is None:
            detection_qualities = [1.0] * len(image_pts)
            
        # Add observations to this perspective's buffer
        for img_pt, field_pt, conf, qual in zip(image_pts, field_pts, confidences, detection_qualities):
            obs = PerspectiveObservation(
                image_pt=img_pt,
                field_pt=field_pt, 
                frame_id=frame_id,
                camera_angle=camera_angle,
                confidence=conf,
                detection_quality=qual
            )
            self.observations_by_angle[camera_angle].append(obs)
            
        # Update individual homography for this angle
        self._update_angle_homography(camera_angle)
        
        # Update master homography combining all angles
        return self._update_master_homography(frame_id)
    
    def _update_angle_homography(self, camera_angle: CameraAngle) -> bool:
        """Update homography for a specific camera angle"""
        
        obs_list = list(self.observations_by_angle[camera_angle])
        if len(obs_list) < self.min_points_per_homography:
            return False
            
        # Extract points
        image_pts = np.array([obs.image_pt for obs in obs_list], dtype=np.float32)
        field_pts = np.array([obs.field_pt for obs in obs_list], dtype=np.float32)
        
        try:
            H, mask = cv2.findHomography(
                image_pts, field_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0,
                confidence=0.80
            )
            
            if H is not None:
                self.homographies_by_angle[camera_angle] = H
                
                # Update coverage map for this angle
                self._update_coverage_map(camera_angle, field_pts[mask.ravel().astype(bool)])
                return True
                
        except Exception as e:
            print(f"Failed to compute homography for {camera_angle}: {e}")
            
        return False
    
    def _update_coverage_map(self, camera_angle: CameraAngle, field_pts: np.ndarray):
        """Track which field regions are covered by this camera angle"""
        
        # Create a simple 2D histogram of field coverage
        # Assuming field coordinates are in meters, standard FIFA field: 105m x 68m
        field_width, field_height = 105, 68
        grid_size = 20  # 20x20 grid
        
        # Convert field points to grid indices
        x_indices = np.clip(((field_pts[:, 0] + field_width/2) / field_width * grid_size).astype(int), 
                           0, grid_size-1)
        y_indices = np.clip(((field_pts[:, 1] + field_height/2) / field_height * grid_size).astype(int), 
                           0, grid_size-1)
        
        # Create coverage map
        coverage = np.zeros((grid_size, grid_size))
        for x_idx, y_idx in zip(x_indices, y_indices):
            coverage[y_idx, x_idx] += 1
            
        self.coverage_maps[camera_angle] = coverage
    
    def _update_master_homography(self, current_frame: int) -> bool:
        """Combine observations from all camera angles into master homography"""
        
        # Collect all observations from all angles
        all_observations = []
        for angle, obs_deque in self.observations_by_angle.items():
            all_observations.extend(list(obs_deque))
            
        if len(all_observations) < self.min_points_per_homography:
            return False
            
        # Extract points and compute weights
        image_pts = np.array([obs.image_pt for obs in all_observations], dtype=np.float32)
        field_pts = np.array([obs.field_pt for obs in all_observations], dtype=np.float32)
        
        # Compute sophisticated weights
        weights = self._compute_observation_weights(all_observations, current_frame)
        
        try:
            # Use weighted RANSAC or regular RANSAC with good parameters
            H, mask = cv2.findHomography(
                image_pts, field_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=8.0,  # More lenient with diverse data
                confidence=0.995,
                maxIters=10000
            )
            
            if H is not None:
                # Optionally: weighted refinement on inliers
                inlier_mask = mask.ravel().astype(bool)
                if inlier_mask.sum() >= 10:
                    # Could do weighted least squares refinement here
                    pass
                    
                self.master_homography = H
                self.last_update_frame = current_frame
                
                print(f"Updated master homography using {inlier_mask.sum()} inliers "
                      f"from {len(set(obs.camera_angle for obs in all_observations))} camera angles")
                return True
                
        except Exception as e:
            print(f"Failed to compute master homography: {e}")
            
        return False
    
    def _compute_observation_weights(self, 
                                   observations: List[PerspectiveObservation], 
                                   current_frame: int) -> np.ndarray:
        """Compute sophisticated weights for each observation"""
        
        weights = np.ones(len(observations))
        
        for i, obs in enumerate(observations):
            weight = 1.0
            
            # 1. Base confidence and detection quality
            weight *= obs.confidence * obs.detection_quality
            
            # 2. Temporal decay (newer observations weighted higher)
            frame_age = current_frame - obs.frame_id
            weight *= (self.perspective_weight_decay ** frame_age)
            
            # 3. Camera angle diversity bonus
            weight *= self._get_angle_diversity_bonus(obs.camera_angle)
            
            # 4. Spatial diversity bonus (boost points in sparse regions)
            weight *= self._get_spatial_diversity_bonus(obs.field_pt, observations)
            
            weights[i] = weight
            
        # Normalize weights
        weights = weights / weights.max()
        return weights
    
    def _get_angle_diversity_bonus(self, camera_angle: CameraAngle) -> float:
        """Give bonus to underrepresented camera angles"""
        
        angle_counts = defaultdict(int)
        for angle, obs_deque in self.observations_by_angle.items():
            angle_counts[angle] = len(obs_deque)
            
        if not angle_counts:
            return 1.0
            
        # Inverse frequency weighting
        total_obs = sum(angle_counts.values())
        angle_frequency = angle_counts[camera_angle] / total_obs
        
        # Boost rare angles
        return 1.0 + (1.0 - angle_frequency)
    
    def _get_spatial_diversity_bonus(self, 
                                   field_pt: Tuple[float, float], 
                                   all_observations: List[PerspectiveObservation]) -> float:
        """Give bonus to points in spatially sparse regions"""
        
        if len(all_observations) < 20:
            return 1.0
            
        # Count nearby points (simple distance-based)
        nearby_count = 0
        for obs in all_observations:
            dist = np.linalg.norm(np.array(field_pt) - np.array(obs.field_pt))
            if dist < 10.0:  # Within 10 meters
                nearby_count += 1
                
        # Boost points in sparse areas
        if nearby_count > 0:
            return 1.0 + self.spatial_diversity_bonus / nearby_count
        else:
            return 1.0 + self.spatial_diversity_bonus
    
    def get_homography(self, 
                      camera_angle: Optional[CameraAngle] = None) -> Optional[np.ndarray]:
        """Get homography - either master or for specific camera angle"""
        
        if camera_angle is None:
            return self.master_homography
        else:
            return self.homographies_by_angle.get(camera_angle)
    
    def get_coverage_analysis(self) -> Dict:
        """Analyze how well different field regions are covered"""
        
        total_coverage = np.zeros((20, 20))
        angle_contributions = {}
        
        for angle, coverage_map in self.coverage_maps.items():
            total_coverage += coverage_map
            angle_contributions[angle.value] = coverage_map.sum()
            
        # Identify poorly covered regions
        poorly_covered = total_coverage < 2  # Less than 2 observations per grid cell
        
        return {
            "total_observations": sum(len(obs_deque) for obs_deque in self.observations_by_angle.values()),
            "angles_used": list(self.observations_by_angle.keys()),
            "angle_contributions": angle_contributions,
            "poorly_covered_ratio": poorly_covered.sum() / poorly_covered.size,
            "homography_available": self.master_homography is not None
        }
    
    def detect_camera_angle(self, 
                          image_pts: List[Tuple[float, float]], 
                          image_shape: Tuple[int, int]) -> CameraAngle:
        """Automatically detect camera angle based on keypoint distribution"""
        
        if not image_pts:
            return CameraAngle.UNKNOWN
            
        img_height, img_width = image_shape
        pts_array = np.array(image_pts)
        
        # Analyze point distribution
        x_center = pts_array[:, 0].mean() / img_width
        y_center = pts_array[:, 1].mean() / img_height
        x_spread = pts_array[:, 0].std() / img_width
        y_spread = pts_array[:, 1].std() / img_height
        
        # Simple heuristics for angle detection
        if x_spread > 0.6 and y_spread > 0.4:
            return CameraAngle.WIDE_CENTER
        elif x_center < 0.3:
            return CameraAngle.LEFT_SIDE
        elif x_center > 0.7:
            return CameraAngle.RIGHT_SIDE
        elif y_center < 0.3 or y_center > 0.7:
            return CameraAngle.BEHIND_GOAL_1 if y_center < 0.5 else CameraAngle.BEHIND_GOAL_2
        else:
            return CameraAngle.UNKNOWN

# Usage example:
def process_multi_perspective_video():
    """Example usage with camera angle detection"""
    
    estimator = MultiPerspectiveHomographyEstimator(
        max_observations_per_angle=300,
        min_points_per_homography=10
    )
    
    for frame_id in range(1000):
        # Your keypoint detection
        image_pts, field_pts = detect_keypoints_for_frame(frame_id)
        
        if image_pts:
            # Auto-detect camera angle (or use manual detection/metadata)
            camera_angle = estimator.detect_camera_angle(image_pts, (1080, 1920))
            
            # Add observations
            success = estimator.add_perspective_keypoints(
                image_pts, field_pts, frame_id, camera_angle
            )
            
            if success:
                # Get the best homography (master combining all angles)
                H_master = estimator.get_homography()
                
                # Get coverage analysis
                analysis = estimator.get_coverage_analysis()
                
                print(f"Frame {frame_id}: {camera_angle.value} | "
                      f"Total obs: {analysis['total_observations']} | "
                      f"Coverage: {(1-analysis['poorly_covered_ratio'])*100:.1f}%")

def detect_keypoints_for_frame(frame_id):
    """Your keypoint detection implementation"""
    return [], []