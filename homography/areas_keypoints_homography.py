import json
import os
import numpy as np
import cv2
import requests
import base64
from inference_sdk import InferenceHTTPClient

keypoints_by_area = {
    "EighteenYardDee": ["TRArc", "TLArc", "BRArc", "BLArc"],
    "EighteenYardBox": ["TR18ML", "TR6ML", "TL6ML", "TL18ML", "TR6MC", "TL6MC", "TPS", "TR18MC", "TRArc", "TLArc", "TL18MC", 
                        "BR18MC", "BRArc", "BLArc", "BL18MC", "BPS", "BR6MC", "BL6MC", "BR18ML", "BR6ML", "BL6ML", "BL18ML"],
    "SixYardBox": ["TR6ML", "TL6ML", "TR6MC", "TL6MC", "BR6MC", "BL6MC", "BR6ML", "BL6ML"],
    "LeftCentreCircle": ["RMC", "LMC", "TCS"],
    "LeftHalf": ["LML", "RML", "TLC", "TRC"],
    "RightCentreCircle": ["RMC", "LMC", "BCS"],
    "RightHalf": ["LML", "RML", "BLC", "BRC"]
}

_area_class_map = None
def load_area_class_map(path="calib/areas_class_map.json"):
    """
    Load the class mapping for area detection
    Maps class_id numbers to semantic area names
    """
    global _area_class_map
    if _area_class_map is None:
        with open(path, "r") as f:
            _area_class_map = json.load(f)
    return _area_class_map

_class_map = None
def load_class_map(path="calib/class_map.json"):
    global _class_map
    if _class_map is None:
        with open(path, "r") as f:
            _class_map = json.load(f)
    return _class_map

# API requesting -----------------

def return_api_request(frame, model_id):
    """
    Call the new football area detection API
    """
    success, buf = cv2.imencode('.jpg', frame)
    if not success:
        raise RuntimeError("JPEG encode failed")
    img_bytes = buf.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=os.getenv("ROBOFLOW_API_KEY")
    )

    result = CLIENT.infer(img_b64, model_id)
    return result

def detect_pitch_keypoints(frame):
    """
    Fetches the raw pitch keypoints via return_api_request(),
    then filters out low‐confidence or edge‐lying points,
    and finally maps each numeric code to its semantic label.
    Returns two parallel lists: (image_pts, labels).
    """
    # 1) call your existing API wrapper
    # data  = return_api_request(frame, "football-field-detection-f07vi/15") COMMENTED OUT
    # data  = return_api_request(frame, "soccer-field-localization/9") COMMENTED OUT
    data  = return_api_request(frame, "football-field-detection-f07vi-1vzgf/5")
    preds = data.get("predictions", [])
    if not preds:
        return [], []

    # 2) extract the keypoints list from the first (pitch) detection
    kp_list = preds[0].get("keypoints", [])
    if not kp_list:
        return [], []

    # 3) prepare for filtering
    class_map = load_class_map()                   # numeric→semantic
    fh, fw    = frame.shape[:2]                    # frame height/width
    keypoint_info = {}

    # 4) loop & filter
    for kp in kp_list:
        conf = kp.get("confidence", 0.0)
        x, y = kp["x"], kp["y"]
        code = kp.get("class")
        # drop low-confidence
        if conf < 0.85:
            continue
        # drop points on the very edge (within 5px)
        if x <= 5 or x >= fw - 5 or y <= 5 or y >= fh - 5:
            continue
        # only keep known labels
        if code not in class_map:
            continue
        # passed all filters ⇒ keep it
        keypoint_info[code] = [x, y]

    return keypoint_info

def detect_area_keypoints(frame):
    """
    Detect football areas and extract corner points for homography calculation
    Returns two parallel lists: (image_pts, labels) where labels are semantic area names
    """
    # 1) Call the area detection API
    data = return_api_request(frame, "football0detections/1")
    predictions = data.get("predictions", [])
    if not predictions:
        return [], []

    # 2) Load class mapping
    class_map = load_area_class_map()  # numeric→semantic
    fh, fw = frame.shape[:2]      # frame height/width
    
    # A dictionary with key of area name and values of a list of: x, y coord of centre and height and width of box
    area_info = {}

    # 3) Process each detected area
    for detection in predictions:
        confidence_score = detection.get("confidence", 0.0)
        class_id = str(detection.get("class_id", ""))
        
        # Filter by confidence
        if confidence_score < 0.70:  # Lower threshold since you had filtering issues
            continue
            
        # Check if we have a mapping for this class
        if class_id not in class_map:
            print(f"Warning: class_id '{class_id}' not found in class_map")
            continue
            
        area_name = class_map[class_id]
        
        # Get bounding box info
        center_x = detection["x"]
        center_y = detection["y"]
        width = detection["width"]
        height = detection["height"]
        
        area_info[area_name] = [center_x, center_y, width, height]
        
        print(f"Added {area_name} with confidence {confidence_score:.2f}")

    return area_info

def match_keypoints_to_areas(frame):
    """
    Detects pitch keypoints and areas, then matches keypoints to their corresponding areas.
    Uses a 10% buffer around each area's boundaries for more flexible matching.
    Keypoints belonging to LeftHalf/RightHalf are excluded from other area detections.
    
    Returns:
        image_pts: list of [x, y] coordinates 
        labels: list of keypoint labels (corresponding to image_pts)
    """
    # 1. Call to detect pitch and area keypoints
    keypoint_info = detect_pitch_keypoints(frame)  # {code: [x, y]}
    area_info = detect_area_keypoints(frame)       # {area_name: [center_x, center_y, width, height]}
    
    # 2. First, identify keypoints that belong to LeftHalf/RightHalf areas
    half_keypoints = set()  # Store keypoint codes that belong to half areas
    class_map = load_class_map()  # Load the numeric code to semantic label mapping
    
    # Get semantic labels for LeftHalf and RightHalf keypoints
    left_half_labels = set(keypoints_by_area.get("LeftHalf", []))
    right_half_labels = set(keypoints_by_area.get("RightHalf", []))
    half_area_labels = left_half_labels | right_half_labels
    
    # Find keypoint codes that correspond to half area labels
    for keypoint_code in keypoint_info.keys():
        if keypoint_code in class_map:
            semantic_label = class_map[keypoint_code]
            if semantic_label in half_area_labels:
                half_keypoints.add(keypoint_code)
    
    # 3. For each area, check which keypoints are within its boundaries (with 10% buffer)
    area_keypoints = {}  # {area_name: {keypoint_label: [x, y]}}
    
    for area_name, area_data in area_info.items():
        center_x, center_y, width, height = area_data
        
        # Add 10% buffer to width and height
        buffered_width = width * 1.1
        buffered_height = height * 1.1
        
        # Calculate area boundaries with buffer (rectangular bounding box)
        left = center_x - buffered_width / 2
        right = center_x + buffered_width / 2
        top = center_y - buffered_height / 2
        bottom = center_y + buffered_height / 2
        
        area_keypoints[area_name] = {}
        
        # Check each detected keypoint
        for keypoint_code, coords in keypoint_info.items():
            x, y = coords
            
            # EXCLUSION RULE: If this is a half-area keypoint and we're not processing a half area, skip it
            if keypoint_code in half_keypoints and area_name not in ["LeftHalf", "RightHalf"]:
                continue
            
            # Check if keypoint is within this area's buffered boundaries
            if left <= x <= right and top <= y <= bottom:
                area_keypoints[area_name][keypoint_code] = [x, y]
    
    # 4. Build final lists - check if detected keypoints are valid for their areas
    image_pts = []
    labels = []
    used_keypoints = set()  # To ensure each keypoint is only added once
    
    for area_name, detected_keypoints in area_keypoints.items():
        # Get the list of keypoints that should belong to this area
        if area_name not in keypoints_by_area:
            continue
            
        valid_keypoints_for_area = keypoints_by_area[area_name]
        
        for keypoint_code, coords in detected_keypoints.items():
            # Convert numeric code to semantic label
            if keypoint_code not in class_map:
                continue
            semantic_label = class_map[keypoint_code]
            
            # Check if this semantic label should belong to this area AND hasn't been used yet
            if semantic_label in valid_keypoints_for_area and semantic_label not in used_keypoints:
                image_pts.append((coords[0], coords[1]))  # Convert to tuple format
                labels.append(semantic_label)
                used_keypoints.add(semantic_label)
    
    return image_pts, labels

# Computing Homography ----------------------------------

def compute_homography(src_pts, dst_pts):
    """
    Enhanced compute_homography with better error handling
    """
    try:
        if len(src_pts) < 4:
            raise RuntimeError(f"Need at least 4 points for homography, got {len(src_pts)}")
            
        # Convert to numpy arrays once at the beginning
        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        
        # Additional validation
        if len(src_pts) != len(dst_pts):
            raise RuntimeError(f"Mismatch in point counts. Source: {len(src_pts)}, Destination: {len(dst_pts)}")
        
        # Try refined homography first
        H, inlier_mask = refine_homography(src, dst,
                                       ransac_thresh=8.0,
                                       reproj_thresh=5.0,
                                       max_iter=5)

        if H is None:
            print("Warning: Refined homography failed, trying basic RANSAC")
            # Fallback to basic RANSAC - use numpy arrays, not original lists
            H, mask = cv2.findHomography(src, dst,  # Use src/dst arrays, not src_pts/dst_pts
                                         method=cv2.RANSAC, 
                                         ransacReprojThreshold=8.0)
            if H is None:
                print("Warning: Basic RANSAC failed, trying least squares")
                # Last resort: least squares (no outlier rejection)
                H, mask = cv2.findHomography(src, dst, method=0)  # Use arrays here too
                if H is None:
                    raise RuntimeError("All homography computation methods failed")
        
        # Validate the computed homography
        if H is not None:
            # Check if homography is reasonable (not degenerate)
            det = np.linalg.det(H[:2, :2])  # Check determinant of top-left 2x2
            if abs(det) < 1e-6:
                print("Warning: Homography appears to be degenerate (near-zero determinant)")
                return None
        
        return H

    except Exception as e:
        print(f"Error in compute_homography: {str(e)}")
        raise RuntimeError(f"Failed to compute homography: {str(e)}")


def refine_homography(src_pts, dst_pts,
                      ransac_thresh,
                      reproj_thresh,
                      max_iter):
    """
    src_pts: numpy array of (x,y) in image
    dst_pts: numpy array of (x,y) in map
    ransac_thresh: inlier threshold for initial RANSAC
    reproj_thresh: max reprojection error (px) to keep a point
    max_iter: how many refine rounds to do
    Returns: final H, inlier_mask (or None, None if failed)
    """
    try:
        # Input should already be numpy arrays from compute_homography
        src = src_pts.astype(np.float32)
        dst = dst_pts.astype(np.float32)

        # Validate input points
        if len(src) < 4 or len(dst) < 4:
            print(f"Warning: Need at least 4 points for homography. Got {len(src)} source and {len(dst)} destination points.")
            return None, None

        if len(src) != len(dst):
            print(f"Warning: Mismatch in point counts. Source: {len(src)}, Destination: {len(dst)}")
            return None, None

        # Check for degenerate cases (all points collinear, etc.)
        if len(src) == 4:
            # Check if points are collinear
            def are_collinear(pts):
                if len(pts) < 3:
                    return False
                # Check if all points lie on the same line
                p1, p2, p3 = pts[0], pts[1], pts[2]
                # Cross product should be zero for collinear points
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                return abs(cross) < 1e-3  # Slightly less strict threshold

            if are_collinear(src) or are_collinear(dst):
                print("Warning: Points appear to be collinear, cannot compute homography")
                return None, None

        # 1) Initial RANSAC fit
        H, mask = cv2.findHomography(src, dst,  # Use numpy arrays
                                     method=cv2.RANSAC,
                                     ransacReprojThreshold=ransac_thresh)
        
        if H is None:
            print(f"Warning: Initial RANSAC failed with {len(src)} points and threshold {ransac_thresh}")
            # Try with a more lenient threshold
            H, mask = cv2.findHomography(src, dst,
                                         method=cv2.RANSAC,
                                         ransacReprojThreshold=ransac_thresh * 2)
            if H is None:
                print("Warning: Even lenient RANSAC failed, trying least squares method")
                # Try least squares as last resort
                H, mask = cv2.findHomography(src, dst, method=0)
                if H is None:
                    print("Error: All homography computation methods failed")
                    return None, None

        if mask is None:
            # Create a mask of all True if findHomography didn't return one
            mask = np.ones((len(src), 1), dtype=np.uint8)

        mask = mask.ravel().astype(bool)

        # Refinement iterations
        for i in range(max_iter):
            try:
                # 2) Compute reprojection of src -> dst_est
                src_in  = src [mask]
                dst_in  = dst [mask]
                
                if len(src_in) < 4:
                    print(f"Warning: Only {len(src_in)} inlier points remaining, stopping refinement")
                    break
                
                dst_est = cv2.perspectiveTransform(src_in.reshape(-1,1,2), H)
                dst_est = dst_est.reshape(-1,2)

                # 3) Compute reprojection error
                errs = np.linalg.norm(dst_est - dst_in, axis=1)

                # 4) Keep only those below reproj_thresh
                good = errs < reproj_thresh
                if good.sum() < 4:
                    print(f"Warning: Only {good.sum()} points below reprojection threshold, stopping refinement")
                    break  # not enough points to refit

                # If no change in inliers, we've converged
                if good.sum() == mask.sum():
                    break

                # 5) Refit H on the "good" subset
                src_fit = src_in[good]
                dst_fit = dst_in[good]
                new_H, new_mask = cv2.findHomography(src_fit, dst_fit,
                                                     method=0)  # regular least-squares
                if new_H is None:
                    print(f"Warning: Refinement iteration {i+1} failed, keeping previous H")
                    break
                
                H = new_H

                # Build new overall mask
                # First start all false, then set true for those in this good set
                new_full_mask = np.zeros_like(mask)
                inds = np.where(mask)[0]
                new_full_mask[inds[good]] = True
                mask = new_full_mask

            except Exception as e:
                print(f"Warning: Error during refinement iteration {i+1}: {str(e)}")
                break

        return H, mask

    except Exception as e:
        print(f"Error in refine_homography: {str(e)}")
        return None, None

# Warping Points ------------------------------------

def warp_point(pt, H):
    """
    pt: (x, y) pixel in image space
    H:  3x3 homography matrix
    returns: (x', y') in map/image coordinates
    """
    src = np.array([[pt]], dtype=np.float32)       # shape (1,1,2)
    dst = cv2.perspectiveTransform(src, H)         # shape (1,1,2)
    return float(dst[0,0,0]), float(dst[0,0,1])

def warp_points(pts, H):
    """
    pts: list of (x, y) tuples in image space
    H:   3x3 homography matrix

    returns: list of (x', y') warped coordinates
    """
    if not pts:
        return []

    # Convert to shape (N,1,2) for perspectiveTransform
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(arr, H)  # returns (N,1,2)
    # Flatten back to list of tuples
    return [(float(p[0]), float(p[1])) for p in warped.reshape(-1, 2)]