# Homography Diagnostic Tool

import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
from homography.homography import detect_pitch_keypoints, compute_homography, warp_point

def diagnose_homography(frame, map_img, map_keypoints, debug_dir="debug_homography"):
    """
    Diagnose homography issues by:
    1. Visualizing detected keypoints on frame
    2. Visualizing corresponding map keypoints
    3. Computing and testing homography with different methods
    4. Visualizing the warping results
    """
    # Create debug directory
    os.makedirs(debug_dir, exist_ok=True)
    
    # 1. Detect keypoints
    image_pts, labels = detect_pitch_keypoints(frame)
    
    # Create a copy of frame for visualization
    debug_frame = frame.copy()
    
    # Plot all keypoints detected
    for i, ((x, y), label) in enumerate(zip(image_pts, labels)):
        cv2.circle(debug_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(debug_frame, f"{i}: {label}", (int(x) + 5, int(y) + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite(f"{debug_dir}/detected_keypoints_all.jpg", debug_frame)
    
    # 2. Filter to only known keypoints in map
    debug_frame_filtered = frame.copy()
    map_img_debug = map_img.copy()
    
    filtered = []
    for i, ((x, y), label) in enumerate(zip(image_pts, labels)):
        if label in map_keypoints:
            filtered.append((x, y, label))
            
            # Draw on frame
            cv2.circle(debug_frame_filtered, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(debug_frame_filtered, f"{i}: {label}", (int(x) + 5, int(y) + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw corresponding point on map
            map_x, map_y = map_keypoints[label]
            cv2.circle(map_img_debug, (int(map_x), int(map_y)), 5, (0, 0, 255), -1)
            cv2.putText(map_img_debug, f"{i}: {label}", (int(map_x) + 5, int(map_y) + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(f"{debug_dir}/filtered_keypoints.jpg", debug_frame_filtered)
    cv2.imwrite(f"{debug_dir}/map_keypoints.jpg", map_img_debug)
    
    # 3. Compute homography if enough points
    if len(filtered) >= 4:
        src_pts = np.array([(x, y) for x, y, _ in filtered], dtype=np.float32)
        dst_pts = np.array([map_keypoints[label] for _, _, label in filtered], dtype=np.float32)
        
        # Save points for reference
        with open(f"{debug_dir}/homography_points.txt", "w") as f:
            f.write("Source Points (image):\n")
            for i, (x, y, label) in enumerate(filtered):
                f.write(f"{i}: {label} -> ({x}, {y})\n")
            
            f.write("\nDestination Points (map):\n")
            for i, (_, _, label) in enumerate(filtered):
                mx, my = map_keypoints[label]
                f.write(f"{i}: {label} -> ({mx}, {my})\n")
        
        # Try computing homography and test it
        try:
            # Regular OpenCV homography
            H_cv, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Your custom homography function
            H_custom = compute_homography(src_pts.tolist(), dst_pts.tolist())
            
            # Compare the two homographies
            print("OpenCV Homography:")
            print(H_cv)
            print("\nCustom Homography:")
            print(H_custom)
            
            # Write homography matrices to file
            with open(f"{debug_dir}/homography_matrices.txt", "w") as f:
                f.write("OpenCV Homography:\n")
                f.write(str(H_cv))
                f.write("\n\nCustom Homography:\n")
                f.write(str(H_custom))
            
            # Test both homographies on a grid of points
            test_warp_grid(frame, map_img, H_cv, "opencv", debug_dir)
            test_warp_grid(frame, map_img, H_custom, "custom", debug_dir)
            
            # Test both homographies on the detected keypoints
            test_warp_keypoints(filtered, map_keypoints, H_cv, "opencv", debug_dir)
            test_warp_keypoints(filtered, map_keypoints, H_custom, "custom", debug_dir)
            
            # Test INVERSE homography (map -> frame)
            # This will tell us if we need to invert the order of src/dst points
            H_inverse, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            test_warp_grid(map_img, frame, H_inverse, "inverse", debug_dir)
            
            return {
                "status": "success",
                "cv_homography": H_cv,
                "custom_homography": H_custom,
                "num_points": len(filtered)
            }
        
        except Exception as e:
            print(f"Error computing homography: {e}")
            with open(f"{debug_dir}/error.txt", "w") as f:
                f.write(f"Error computing homography: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "num_points": len(filtered)
            }
    else:
        print(f"Not enough keypoints (only {len(filtered)})")
        with open(f"{debug_dir}/error.txt", "w") as f:
            f.write(f"Not enough keypoints (only {len(filtered)})")
        
        return {
            "status": "insufficient",
            "num_points": len(filtered)
        }

def test_warp_grid(src_img, dst_img, H, label, debug_dir):
    """Test homography by warping a grid of points and visualizing results."""
    h, w = src_img.shape[:2]
    dh, dw = dst_img.shape[:2]
    
    # Create a grid of points in source image
    grid_step = 50
    grid_pts = []
    for y in range(0, h, grid_step):
        for x in range(0, w, grid_step):
            grid_pts.append((x, y))
    
    # Convert to array for warping
    grid_array = np.array(grid_pts, dtype=np.float32).reshape(-1, 1, 2)
    
    # Warp points
    try:
        warped_pts = cv2.perspectiveTransform(grid_array, H)
        warped_pts = warped_pts.reshape(-1, 2)
        
        # Create visualization
        dst_copy = dst_img.copy()
        
        # Draw warped grid points
        for pt in warped_pts:
            x, y = pt
            if 0 <= x < dw and 0 <= y < dh:  # Only if point is within bounds
                cv2.circle(dst_copy, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        cv2.imwrite(f"{debug_dir}/grid_warp_{label}.jpg", dst_copy)
        
        # Count points that fall outside the destination image
        out_of_bounds = sum(1 for pt in warped_pts if not (0 <= pt[0] < dw and 0 <= pt[1] < dh))
        with open(f"{debug_dir}/grid_warp_{label}_stats.txt", "w") as f:
            f.write(f"Total points: {len(grid_pts)}\n")
            f.write(f"Out of bounds: {out_of_bounds} ({out_of_bounds/len(grid_pts)*100:.1f}%)\n")
        
        return True
    except Exception as e:
        print(f"Error warping grid: {e}")
        return False

def test_warp_keypoints(filtered_pts, map_keypoints, H, label, debug_dir):
    """Test how well the homography maps the actual keypoints."""
    src_pts = np.array([(x, y) for x, y, _ in filtered_pts], dtype=np.float32).reshape(-1, 1, 2)
    
    # Expected destinations
    expected_dst_pts = np.array([map_keypoints[lbl] for _, _, lbl in filtered_pts], dtype=np.float32)
    
    # Actual destinations after warping
    try:
        actual_dst_pts = cv2.perspectiveTransform(src_pts, H).reshape(-1, 2)
        
        # Calculate errors
        errors = np.sqrt(np.sum((expected_dst_pts - actual_dst_pts)**2, axis=1))
        
        # Write error report
        with open(f"{debug_dir}/keypoint_errors_{label}.txt", "w") as f:
            f.write("Keypoint Warping Errors:\n")
            for i, ((_, _, lbl), err) in enumerate(zip(filtered_pts, errors)):
                src_x, src_y = filtered_pts[i][0], filtered_pts[i][1]
                exp_x, exp_y = map_keypoints[lbl]
                act_x, act_y = actual_dst_pts[i]
                
                f.write(f"{i}: {lbl} -> Expected ({exp_x:.1f}, {exp_y:.1f}), Got ({act_x:.1f}, {act_y:.1f}), Error: {err:.1f} pixels\n")
            
            f.write(f"\nAverage Error: {np.mean(errors):.2f} pixels\n")
            f.write(f"Max Error: {np.max(errors):.2f} pixels\n")
        
        return True
    except Exception as e:
        print(f"Error testing keypoints: {e}")
        return False

def analyze_map_keypoints(map_keypoints_path, debug_dir="debug_homography"):
    """Analyze map keypoints dictionary structure and visualize points."""
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load map keypoints
    with open(map_keypoints_path, "r") as f:
        map_keypoints = json.load(f)
    
    # Basic statistics
    with open(f"{debug_dir}/map_keypoints_analysis.txt", "w") as f:
        f.write(f"Total keypoints in map: {len(map_keypoints)}\n\n")
        f.write("Keypoint Labels and Coordinates:\n")
        
        # Write all keypoints
        for label, coords in map_keypoints.items():
            f.write(f"{label}: {coords}\n")
        
        # Check for potential issues
        f.write("\nPotential Issues:\n")
        
        # Check if map points are all within a reasonable range
        all_x = [coords[0] for coords in map_keypoints.values()]
        all_y = [coords[1] for coords in map_keypoints.values()]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        f.write(f"X range: {min_x} to {max_x}\n")
        f.write(f"Y range: {min_y} to {max_y}\n")
        
        # Check if any coordinates are negative
        if min_x < 0 or min_y < 0:
            f.write("WARNING: Some map coordinates are negative!\n")
        
        # Check if any points have the same coordinates
        duplicate_coords = {}
        for label, coords in map_keypoints.items():
            coords_tuple = tuple(coords)
            if coords_tuple in duplicate_coords:
                duplicate_coords[coords_tuple].append(label)
            else:
                duplicate_coords[coords_tuple] = [label]
        
        duplicates = {coords: labels for coords, labels in duplicate_coords.items() if len(labels) > 1}
        if duplicates:
            f.write("\nWARNING: Multiple labels have the same coordinates:\n")
            for coords, labels in duplicates.items():
                f.write(f"  {coords}: {', '.join(labels)}\n")
        
        # Check if any points are very close to each other
        threshold = 5  # pixels
        close_points = []
        
        for label1, coords1 in map_keypoints.items():
            for label2, coords2 in map_keypoints.items():
                if label1 != label2:
                    dist = np.sqrt((coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2)
                    if dist < threshold:
                        close_points.append((label1, label2, dist))
        
        if close_points:
            f.write(f"\nWARNING: Some points are very close to each other (< {threshold} pixels):\n")
            for label1, label2, dist in close_points:
                f.write(f"  {label1} and {label2}: {dist:.2f} pixels apart\n")
    
    # Create visualization of map keypoints
    try:
        # Create a blank canvas
        width = int(max_x + 100)
        height = int(max_y + 100)
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw points
        for label, (x, y) in map_keypoints.items():
            cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(canvas, label, (int(x + 5), int(y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(f"{debug_dir}/map_keypoints_visualization.jpg", canvas)
    except Exception as e:
        print(f"Error visualizing map keypoints: {e}")

def get_homography_validation_img(map_img_path, debug_dir="debug_homography"):
    """
    Creates a validation image with pitch landmark labels for checking
    against the keypoints dataset.
    """
    os.makedirs(debug_dir, exist_ok=True)
    
    try:
        # Load map image
        map_img = cv2.imread(map_img_path)
        if map_img is None:
            print(f"Could not load map image: {map_img_path}")
            return
        
        cv2.imwrite(f"{debug_dir}/original_map.jpg", map_img)
        
        # Create labeled grid on map image for reference
        h, w = map_img.shape[:2]
        grid_img = map_img.copy()
        
        # Draw grid lines
        grid_step = 50
        for y in range(0, h, grid_step):
            cv2.line(grid_img, (0, y), (w, y), (200, 200, 200), 1)
            cv2.putText(grid_img, str(y), (5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        for x in range(0, w, grid_step):
            cv2.line(grid_img, (x, 0), (x, h), (200, 200, 200), 1)
            cv2.putText(grid_img, str(x), (x+5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imwrite(f"{debug_dir}/map_with_grid.jpg", grid_img)
        
    except Exception as e:
        print(f"Error creating validation image: {e}")

def run_diagnostics():
    """
    Main function to run all diagnostics.
    """
    # Create debug directory
    debug_dir = "debug_homography"
    os.makedirs(debug_dir, exist_ok=True)
    
    # 1. Analyze map keypoints file
    analyze_map_keypoints("calib/map_keypoints.json", debug_dir)
    
    # 2. Create validation image with grid
    get_homography_validation_img("calib/map.jpg", debug_dir)
    
    print(f"Diagnostic files created in {debug_dir}")
    print("Please check these files to identify homography issues")

if __name__ == "__main__":
    run_diagnostics()