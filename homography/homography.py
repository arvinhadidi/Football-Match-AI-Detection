# homography/homography.py

import json
import os
import numpy as np
import cv2
import requests
import base64
from inference_sdk import InferenceHTTPClient

# Your model details
PROJECT_ID = "roboflow-jvuqo/football-field-detection-f07vi/15"
MODEL_VERSION = 1
confidence = 0.5
iou_thresh = 0.5
API_KEY = os.getenv("ROBOFLOW_API_KEY")  # Make sure this is set

def return_api_request(frame):

    success, buf = cv2.imencode('.jpg', frame)
    if not success:
        raise RuntimeError("JPEG encode failed")
    img_bytes = buf.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    CLIENT = InferenceHTTPClient(
        api_url = "https://serverless.roboflow.com",
        api_key = os.getenv("ROBOFLOW_API_KEY")
    )

    result = CLIENT.infer(img_b64, model_id="football-field-detection-f07vi/14")
    # result = CLIENT.infer(img_b64, model_id="keypoints-on-soccer-pitch/1")
    # print(result)
    return result

_class_map = None
def load_class_map(path="calib/class_map.json"):
    global _class_map
    if _class_map is None:
        with open(path, "r") as f:
            _class_map = json.load(f)
    return _class_map

# ------------------------------------
# 2) Detect pitch keypoints in a frame
# ------------------------------------
# def detect_pitch_keypoints(frame):
#     data = return_api_request(frame)
#     preds = data.get("predictions", [])
#     if not preds:
#         return [], []

#     # Take the first (and only) pitch detection
#     kp_list = preds[0].get("keypoints", [])
#     if not kp_list:
#         return [], []

#     class_map = load_class_map()
#     image_pts, labels = [], []

#     for kp in kp_list:
#         code = kp.get("class")
#         if code in class_map:
#             image_pts.append((kp["x"], kp["y"]))
#             labels.append(class_map[code])

#     return image_pts, labels

def detect_pitch_keypoints(frame):
    """
    Fetches the raw pitch keypoints via return_api_request(),
    then filters out low‐confidence or edge‐lying points,
    and finally maps each numeric code to its semantic label.
    Returns two parallel lists: (image_pts, labels).
    """
    # 1) call your existing API wrapper
    data  = return_api_request(frame)
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
    image_pts = []
    labels    = []

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
        image_pts.append((x, y))
        labels.append(class_map[code])

    return image_pts, labels



# ------------------------------------
# 3) Compute homography from two point lists
# ------------------------------------
def compute_homography(src_pts, dst_pts):
    src = np.ascontiguousarray(src_pts, dtype=np.float32)
    dst = np.ascontiguousarray(dst_pts, dtype=np.float32)
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    # H, inlier_mask = refine_homography(src, dst,
    #                                ransac_thresh=3.0,
    #                                reproj_thresh=5.0,
    #                                max_iter=5)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
    # if H is None:
    #     raise RuntimeError("RANSAC failed, check correspondences")
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

        # If no change in inliers, we’ve converged
        if good.sum() == mask.sum():
            break

        # 5) Refit H on the “good” subset
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
