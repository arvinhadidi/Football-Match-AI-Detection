# homography/homography.py

import json
import os
import numpy as np
import cv2
import requests
import base64
from inference_sdk import InferenceHTTPClient

# Your model details
PROJECT_ID = "football0detections/1"
API_KEY = os.getenv("ROBOFLOW_API_KEY")  # Make sure this is set

def return_api_request(frame):
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

    result = CLIENT.infer(img_b64, model_id=PROJECT_ID)
    return result

_class_map = None
def load_class_map(path="calib/areas_class_map.json"):
    """
    Load the class mapping for area detection
    Maps class_id numbers to semantic area names
    """
    global _class_map
    if _class_map is None:
        with open(path, "r") as f:
            _class_map = json.load(f)
    return _class_map

_area_keypoints = None
def load_area_keypoints(path="calib/area_keypoints.json"):
    """
    Load the 2D pitch coordinates for each area type
    """
    global _area_keypoints
    if _area_keypoints is None:
        with open(path, "r") as f:
            _area_keypoints = json.load(f)
    return _area_keypoints

def labels_to_pitch_points(labels):
    """
    Convert semantic area labels to corresponding 2D pitch coordinate points
    Each area generates 4 corner points
    """
    area_keypoints = load_area_keypoints()
    pitch_pts = []
    
    # Define approximate dimensions for each area type in pitch coordinates
    area_dimensions = {
        "SixYardBox": (72, 22),        # width, height in pitch units
        "EighteenYardBox": (138, 60),   # width, height in pitch units  
        "EighteenYardDee": (60, 22),   # width, height in pitch units
        "LeftCentreCircle": (70, 36),  # width, height in pitch units
        "RightCentreCircle": (70, 36)  # width, height in pitch units
    }
    
    for area_name in labels:
        # Get center coordinates from area_keypoints.json
        if area_name not in area_keypoints:
            print(f"Warning: No 2D coordinates found for area '{area_name}'")
            continue
            
        center_x, center_y = area_keypoints[area_name]
        
        # Get dimensions
        if area_name not in area_dimensions:
            width, height = (20, 20)  # default
        else:
            width, height = area_dimensions[area_name]
        
        half_w, half_h = width / 2, height / 2
        
        # Generate corner coordinates: top-left, top-right, bottom-right, bottom-left
        corner = (center_x - half_w, center_y - half_h)  # This matches the image corner ordering
        pitch_pts.append(corner)
    
    return pitch_pts

def detect_pitch_keypoints(frame):
    """
    Detect football areas and extract corner points for homography calculation
    Returns two parallel lists: (image_pts, labels) where labels are semantic area names
    """
    # 1) Call the area detection API
    data = return_api_request(frame)
    predictions = data.get("predictions", [])
    if not predictions:
        return [], []

    # 2) Load class mapping
    class_map = load_class_map()  # numeric→semantic
    fh, fw = frame.shape[:2]      # frame height/width
    
    image_pts = []
    labels = []

    # 3) Process each detected area
    for detection in predictions:
        confidence_score = detection.get("confidence", 0.0)
        class_id = str(detection.get("class_id", ""))
        
        # Filter by confidence
        if confidence_score < 0.7:  # Lower threshold since you had filtering issues
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
        
        # Filter out detections too close to frame edges
        margin = 10
        if (center_x - width/2 <= margin or center_x + width/2 >= fw - margin or
            center_y - height/2 <= margin or center_y + height/2 >= fh - margin):
            continue
        
        # Calculate corner points in image coordinates
        half_w, half_h = width / 2, height / 2
        image_corners = [
            (center_x - half_w, center_y - half_h),  # top-left
            (center_x + half_w, center_y - half_h),  # top-right
            (center_x + half_w, center_y + half_h),  # bottom-right
            (center_x - half_w, center_y + half_h)   # bottom-left
        ]
        
        # Add all 4 corners to our point lists with the same semantic label
        image_pts.extend(image_corners)
        labels.extend([area_name + "_TL"])
        labels.extend([area_name + "_TR"])
        labels.extend([area_name + "_BL"])
        labels.extend([area_name + "_TR"])
        
        print(f"Added {area_name} with confidence {confidence_score:.2f}")

    print(f"Total points for homography: {len(image_pts)}")
    print(f"Unique area types: {set(labels)}")
    return image_pts, labels

# ------------------------------------
# 3) Compute homography from two point lists
# ------------------------------------
def compute_homography(src_pts, dst_pts):
    if len(src_pts) < 4:
        raise RuntimeError(f"Need at least 4 points for homography, got {len(src_pts)}")
        
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    
    # H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    H, inlier_mask = refine_homography(src, dst,
                                   ransac_thresh=3.0,
                                   reproj_thresh=5.0,
                                   max_iter=5)

    if H is None:
        raise RuntimeError("RANSAC failed to compute homography")
    
    return H

def refine_homography(src_pts, dst_pts,
                      ransac_thresh=3.0,
                      reproj_thresh=5.0,
                      max_iter=3):
    """
    src_pts: list of (x,y) in image
    dst_pts: list of (x,y) in map
    ransac_thresh: inlier threshold for initial RANSAC
    reproj_thresh: max reprojection error (px) to keep a point
    max_iter: how many refine rounds to do
    Returns: final H, inlier_mask
    """
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    # 1) Initial RANSAC fit
    H, mask = cv2.findHomography(src_pts, dst_pts,
                                 method=cv2.RANSAC,
                                 ransacReprojThreshold=ransac_thresh)
    if H is None:
        raise RuntimeError("Could not compute initial homography")

    mask = mask.ravel().astype(bool)

    for i in range(max_iter):
        # 2) Compute reprojection of src -> dst_est
        src_in  = src [mask]
        dst_in  = dst [mask]
        dst_est = cv2.perspectiveTransform(src_in.reshape(-1,1,2), H)
        dst_est = dst_est.reshape(-1,2)

        # 3) Compute reprojection error
        errs = np.linalg.norm(dst_est - dst_in, axis=1)

        # 4) Keep only those below reproj_thresh
        good = errs < reproj_thresh
        if good.sum() < 4:
            break  # not enough points to refit

        # If no change in inliers, we've converged
        if good.sum() == mask.sum():
            break

        # 5) Refit H on the "good" subset
        src_fit = src_in[good]
        dst_fit = dst_in[good]
        H, new_mask = cv2.findHomography(src_fit, dst_fit,
                                         method=0)  # regular least-squares
        if H is None:
            break

        # Build new overall mask
        # First start all false, then set true for those in this good set
        new_full_mask = np.zeros_like(mask)
        inds = np.where(mask)[0]
        new_full_mask[inds[good]] = True
        mask = new_full_mask

    return H, mask

# ------------------------------------
# 4) Warp any single point (x,y) using H
# ------------------------------------
def warp_point(pt, H):
    """
    pt: (x, y) pixel in image space
    H:  3x3 homography matrix
    returns: (x', y') in map/image coordinates
    """
    src = np.array([[pt]], dtype=np.float32)       # shape (1,1,2)
    dst = cv2.perspectiveTransform(src, H)         # shape (1,1,2)
    return float(dst[0,0,0]), float(dst[0,0,1])

# ------------------------------------
# 5) Warp a list of points at once
# ------------------------------------
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