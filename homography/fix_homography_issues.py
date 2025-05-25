def fix_homography_issues(frame, tracks, map_img, map_keypoints):
    """
    This function attempts several approaches to fix homography issues:
    1. Tries both source→dest and dest→source point ordering
    2. Tests different confidence thresholds
    3. Uses alternative algorithms for homography computation
    4. Performs sanity checking on the resulting matrix
    
    Returns the best homography matrix found and diagnostic information.
    """
    import cv2
    import numpy as np
    from homography.homography import detect_pitch_keypoints, warp_point
    import os
    import json
    
    # Create debug directory
    debug_dir = "debug_homography/fix_attempts"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Storage for all homography results
    results = []
    
    # Try different confidence thresholds
    for conf_threshold in [0.9, 0.8, 0.7, 0.6, 0.5]:
        # 1. Override the detection confidence threshold
        # (This is a workaround - in production you'd modify the detect_pitch_keypoints function)
        old_image_pts, old_labels = detect_pitch_keypoints(frame)
        
        # Filter with our custom threshold (simulating what would happen in detect_pitch_keypoints)
        image_pts, labels = [], []
        
        # Get raw API result to re-filter
        from homography.homography import return_api_request, load_class_map
        data = return_api_request(frame)
        preds = data.get("predictions", [])
        if preds:
            class_map = load_class_map()
            kp_list = preds[0].get("keypoints", [])
            
            # Apply our custom filtering
            fh, fw = frame.shape[:2]
            for kp in kp_list:
                code = kp.get("class")
                conf = kp.get("confidence", 0.0)
                x, y = kp["x"], kp["y"]
                
                # Use our custom threshold
                if conf < conf_threshold:
                    continue
                # Keep other filters as they were
                if x <= 5 or x >= fw - 5 or y <= 5 or y >= fh - 5:
                    continue
                if code not in class_map:
                    continue
                
                image_pts.append((x, y))
                labels.append(class_map[code])
        
        # Skip if too few points
        if len(image_pts) < 4:
            continue
            
        # Filter to keep only known map keypoints
        filtered = []
        for (x, y), lbl in zip(image_pts, labels):
            if lbl in map_keypoints:
                filtered.append((x, y, lbl))
        
        if len(filtered) < 4:
            continue
            
        # Extract source and destination points
        src_pts = np.array([(x, y) for x, y, _ in filtered], dtype=np.float32)
        dst_pts = np.array([map_keypoints[lbl] for _, _, lbl in filtered], dtype=np.float32)
        
        # Try different orderings and methods
        methods = {
            "cv_src_to_dst": {"src": src_pts, "dst": dst_pts, "method": cv2.RANSAC, "thresh": 3.0},
            "cv_dst_to_src": {"src": dst_pts, "dst": src_pts, "method": cv2.RANSAC, "thresh": 3.0},
            "cv_src_to_dst_lmeds": {"src": src_pts, "dst": dst_pts, "method": cv2.LMEDS, "thresh": 3.0},
            "cv_dst_to_src_lmeds": {"src": dst_pts, "dst": src_pts, "method": cv2.LMEDS, "thresh": 3.0},
        }
        
        for method_name, params in methods.items():
            try:
                # Compute homography
                H, mask = cv2.findHomography(
                    params["src"], params["dst"], 
                    method=params["method"], 
                    ransacReprojThreshold=params["thresh"]
                )
                
                if H is None:
                    continue
                    
                # Invert if needed
                if "dst_to_src" in method_name:
                    H = np.linalg.inv(H)
                
                # Sanity check the matrix
                is_sane = True
                # Check for extreme values in H (sign of bad computation)
                if np.max(np.abs(H)) > 1000:
                    is_sane = False
                # Check that determinant is not too small (would cause unstable inverse)
                if np.abs(np.linalg.det(H)) < 1e-6:
                    is_sane = False
                    
                # Calculate reproj error (only for src_to_dst methods)
                if "src_to_dst" in method_name:
                    # Warp source points
                    src_matrix = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
                    warped_pts = cv2.perspectiveTransform(src_matrix, H).reshape(-1, 2)
                    # Calculate error
                    error = np.sqrt(np.sum((warped_pts - dst_pts)**2, axis=1))
                    avg_error = np.mean(error)
                    max_error = np.max(error)
                else:
                    # For dst_to_src methods, calculate error the other way
                    dst_matrix = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
                    warped_pts = cv2.perspectiveTransform(dst_matrix, np.linalg.inv(H)).reshape(-1, 2)
                    error = np.sqrt(np.sum((warped_pts - src_pts)**2, axis=1))
                    avg_error = np.mean(error)
                    max_error = np.max(error)
                
                # Score this attempt
                score = {
                    "method": method_name,
                    "conf_threshold": conf_threshold,
                    "num_points": len(filtered),
                    "is_sane": is_sane,
                    "avg_error": float(avg_error),
                    "max_error": float(max_error),
                    "matrix": H.tolist()
                }
                
                # Add to results if sane
                if is_sane:
                    results.append((score, H))
                    
                # Generate debug grid warping visualization
                debug_frame = frame.copy()
                debug_map = map_img.copy()
                
                # Draw original points in frame
                for i, (x, y, lbl) in enumerate(filtered):
                    cv2.circle(debug_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(debug_frame, lbl, (int(x+5), int(y+5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw destination points in map
                for i, (_, _, lbl) in enumerate(filtered):
                    x, y = map_keypoints[lbl]
                    cv2.circle(debug_map, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(debug_map, lbl, (int(x+5), int(y+5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Warp a grid of points from frame to map
                h, w = frame.shape[:2]
                grid_step = 50
                grid_pts = []
                for y in range(0, h, grid_step):
                    for x in range(0, w, grid_step):
                        grid_pts.append((x, y))
                
                # Create array for warping
                grid_array = np.array(grid_pts, dtype=np.float32).reshape(-1, 1, 2)
                
                try:
                    # Warp points
                    warped_grid = cv2.perspectiveTransform(grid_array, H).reshape(-1, 2)
                    
                    # Draw warped grid points on map
                    mh, mw = map_img.shape[:2]
                    for i, (wx, wy) in enumerate(warped_grid):
                        if 0 <= wx < mw and 0 <= wy < mh:
                            cv2.circle(debug_map, (int(wx), int(wy)), 2, (0, 0, 255), -1)
                    
                    # Save debug images
                    label = f"{method_name}_conf{conf_threshold}"
                    cv2.imwrite(f"{debug_dir}/frame_{label}.jpg", debug_frame)
                    cv2.imwrite(f"{debug_dir}/map_grid_{label}.jpg", debug_map)
                    
                    # Save score info
                    with open(f"{debug_dir}/score_{label}.json", "w") as f:
                        json.dump(score, f, indent=2)
                        
                except Exception as e:
                    print(f"Error warping grid: {e}")
                    
            except Exception as e:
                print(f"Error with {method_name}: {e}")
    
    # Find best result based on error and number of points
    if results:
        # Sort by average error (lower is better)
        results.sort(key=lambda x: x[0]["avg_error"])
        best_score, best_H = results[0]
        
        # Save the best result
        with open(f"{debug_dir}/best_homography.json", "w") as f:
            json.dump(best_score, f, indent=2)
            
        print(f"Best homography method: {best_score['method']}")
        print(f"Confidence threshold: {best_score['conf_threshold']}")
        print(f"Average error: {best_score['avg_error']:.2f} pixels")
        print(f"Number of points: {best_score['num_points']}")
        
        return best_H, best_score
    else:
        print("No valid homography found!")
        return None, None