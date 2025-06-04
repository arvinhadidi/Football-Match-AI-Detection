# Enhanced Homography Diagnostic Tool

import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
from homography.areas_keypoints_homography import (
    detect_pitch_keypoints, 
    detect_area_keypoints, 
    match_keypoints_to_areas,
    compute_homography, 
    warp_point,
    keypoints_by_area,
    load_class_map
)

def create_comprehensive_diagnostic(frame, map_img_path, map_keypoints_path, debug_dir="enhanced_debug"):
    """
    Create comprehensive diagnostic visualizations including:
    1. All detected keypoints with confidence filtering
    2. Detected areas with bounding boxes
    3. Color-coded keypoints (green=correct area, red=incorrect/unmatched)
    4. Homography visualization on 2D pitch
    """
    os.makedirs(debug_dir, exist_ok=True)
    
    # Load map image and keypoints
    map_img = cv2.imread(map_img_path)
    with open(map_keypoints_path, "r") as f:
        map_keypoints = json.load(f)
    
    print("Starting comprehensive diagnostic...")
    
    # 1. Detect all keypoints and areas
    keypoint_info = detect_pitch_keypoints(frame)  # {code: [x, y]}
    area_info = detect_area_keypoints(frame)       # {area_name: [center_x, center_y, width, height]}
    
    print(f"Detected {len(keypoint_info)} keypoints")
    print(f"Detected {len(area_info)} areas")
    
    # 2. Create visualization with all detected keypoints and areas
    frame_vis = create_keypoint_area_visualization(frame, keypoint_info, area_info, debug_dir)
    
    # 3. Match keypoints to areas and create color-coded visualization
    matched_pts, matched_labels = match_keypoints_to_areas(frame)
    color_coded_vis = create_color_coded_visualization(frame, keypoint_info, area_info, 
                                                     matched_pts, matched_labels, debug_dir)
    
    # 4. Create homography visualization if enough points
    if len(matched_pts) >= 4:
        homography_result = create_homography_visualization(
            frame, map_img, matched_pts, matched_labels, map_keypoints, debug_dir
        )
        
        # 5. Create 2D pitch overlay
        create_2d_pitch_overlay(frame, map_img, homography_result['homography'], debug_dir)
        
        return {
            "status": "success",
            "keypoints_detected": len(keypoint_info),
            "areas_detected": len(area_info),
            "matched_points": len(matched_pts),
            "homography": homography_result
        }
    else:
        print(f"Insufficient points for homography: {len(matched_pts)}")
        return {
            "status": "insufficient_points",
            "keypoints_detected": len(keypoint_info),
            "areas_detected": len(area_info),
            "matched_points": len(matched_pts)
        }

def create_keypoint_area_visualization(frame, keypoint_info, area_info, debug_dir):
    """
    Create visualization showing all detected keypoints and area bounding boxes
    """
    vis_frame = frame.copy()
    class_map = load_class_map()
    
    # Draw area bounding boxes first (so they're behind keypoints)
    for area_name, (center_x, center_y, width, height) in area_info.items():
        # Calculate bounding box corners
        left = int(center_x - width / 2)
        right = int(center_x + width / 2)
        top = int(center_y - height / 2)
        bottom = int(center_y + height / 2)
        
        # Draw bounding box
        cv2.rectangle(vis_frame, (left, top), (right, bottom), (255, 255, 0), 2)
        
        # Add area label
        cv2.putText(vis_frame, area_name, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw center point
        cv2.circle(vis_frame, (int(center_x), int(center_y)), 3, (255, 255, 0), -1)
    
    # Draw all detected keypoints
    for keypoint_code, (x, y) in keypoint_info.items():
        # Get semantic label
        semantic_label = class_map.get(keypoint_code, f"Unknown_{keypoint_code}")
        
        # Draw keypoint
        cv2.circle(vis_frame, (int(x), int(y)), 5, (0, 255, 255), -1)  # Yellow
        cv2.putText(vis_frame, semantic_label, (int(x) + 5, int(y) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    cv2.imwrite(f"{debug_dir}/01_all_detections.jpg", vis_frame)
    
    # Create summary text file
    with open(f"{debug_dir}/01_detection_summary.txt", "w") as f:
        f.write("DETECTION SUMMARY\n")
        f.write("================\n\n")
        
        f.write(f"Total keypoints detected: {len(keypoint_info)}\n")
        f.write(f"Total areas detected: {len(area_info)}\n\n")
        
        f.write("DETECTED KEYPOINTS:\n")
        for code, (x, y) in keypoint_info.items():
            label = class_map.get(code, f"Unknown_{code}")
            f.write(f"  {label} ({code}): ({x:.1f}, {y:.1f})\n")
        
        f.write("\nDETECTED AREAS:\n")
        for area_name, (cx, cy, w, h) in area_info.items():
            f.write(f"  {area_name}: center=({cx:.1f}, {cy:.1f}), size=({w:.1f}x{h:.1f})\n")
    
    return vis_frame

def create_color_coded_visualization(frame, keypoint_info, area_info, matched_pts, matched_labels, debug_dir):
    """
    Create visualization with color-coded keypoints:
    - Green: keypoints correctly matched to their expected areas
    - Red: keypoints not matched or in wrong areas
    - Blue: area boundaries
    """
    vis_frame = frame.copy()
    class_map = load_class_map()
    
    # Create set of matched labels for quick lookup
    matched_set = set(matched_labels)
    
    # Draw area boundaries with buffer zones
    for area_name, (center_x, center_y, width, height) in area_info.items():
        # Original boundary
        left = int(center_x - width / 2)
        right = int(center_x + width / 2)
        top = int(center_y - height / 2)
        bottom = int(center_y + height / 2)
        
        # Buffered boundary (10% larger)
        buffered_width = width * 1.1
        buffered_height = height * 1.1
        buff_left = int(center_x - buffered_width / 2)
        buff_right = int(center_x + buffered_width / 2)
        buff_top = int(center_y - buffered_height / 2)
        buff_bottom = int(center_y + buffered_height / 2)
        
        # Draw original boundary in solid blue
        cv2.rectangle(vis_frame, (left, top), (right, bottom), (255, 0, 0), 2)
        
        # Draw buffered boundary in dashed blue (approximate with dotted line)
        cv2.rectangle(vis_frame, (buff_left, buff_top), (buff_right, buff_bottom), (150, 0, 0), 1)
        
        # Area label
        cv2.putText(vis_frame, area_name, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Color-code keypoints
    for keypoint_code, (x, y) in keypoint_info.items():
        semantic_label = class_map.get(keypoint_code, f"Unknown_{keypoint_code}")
        
        # Determine color based on matching status
        if semantic_label in matched_set:
            color = (0, 255, 0)  # Green for correctly matched
            thickness = -1  # Filled
        else:
            color = (0, 0, 255)  # Red for unmatched/incorrect
            thickness = 2   # Outline only
        
        # Draw keypoint
        cv2.circle(vis_frame, (int(x), int(y)), 6, color, thickness)
        
        # Add label with matching color
        cv2.putText(vis_frame, semantic_label, (int(x) + 8, int(y) - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.imwrite(f"{debug_dir}/02_color_coded_keypoints.jpg", vis_frame)
    
    # Create detailed matching report
    with open(f"{debug_dir}/02_matching_report.txt", "w") as f:
        f.write("KEYPOINT MATCHING REPORT\n")
        f.write("========================\n\n")
        
        matched_count = len(matched_set)
        total_count = len(keypoint_info)
        
        f.write(f"Successfully matched: {matched_count}/{total_count} keypoints\n")
        f.write(f"Matching rate: {matched_count/total_count*100:.1f}%\n\n")
        
        f.write("MATCHED KEYPOINTS (GREEN):\n")
        for label in matched_labels:
            f.write(f"  ✓ {label}\n")
        
        f.write("\nUNMATCHED KEYPOINTS (RED):\n")
        for code, (x, y) in keypoint_info.items():
            label = class_map.get(code, f"Unknown_{code}")
            if label not in matched_set:
                f.write(f"  ✗ {label} at ({x:.1f}, {y:.1f})\n")
        
        # Analysis of why keypoints might not match
        f.write("\nMATCHING ANALYSIS:\n")
        f.write("Possible reasons for unmatched keypoints:\n")
        f.write("1. Keypoint is outside any detected area boundaries\n")
        f.write("2. Keypoint is in an area but not expected to be there\n")
        f.write("3. Area detection failed for the keypoint's expected area\n")
        f.write("4. Keypoint label is not in the keypoints_by_area mapping\n")
    
    return vis_frame

def create_homography_visualization(frame, map_img, matched_pts, matched_labels, map_keypoints, debug_dir):
    """
    Create homography and visualize the transformation
    """
    # Filter to only points that exist in map_keypoints
    valid_src_pts = []
    valid_dst_pts = []
    valid_labels = []
    
    for (x, y), label in zip(matched_pts, matched_labels):
        if label in map_keypoints:
            valid_src_pts.append((x, y))
            valid_dst_pts.append(map_keypoints[label])
            valid_labels.append(label)
    
    if len(valid_src_pts) < 4:
        print(f"Insufficient valid points for homography: {len(valid_src_pts)}")
        return {"status": "insufficient", "num_points": len(valid_src_pts)}
    
    try:
        # Compute homography
        H = compute_homography(valid_src_pts, valid_dst_pts)
        
        # Create visualization showing point correspondences
        correspondence_vis = create_correspondence_visualization(
            frame, map_img, valid_src_pts, valid_dst_pts, valid_labels, debug_dir
        )
        
        # Test homography accuracy
        accuracy_report = test_homography_accuracy(valid_src_pts, valid_dst_pts, valid_labels, H, debug_dir)
        
        # Visualize warped grid
        create_warped_grid_visualization(frame, map_img, H, debug_dir)
        
        return {
            "status": "success",
            "homography": H,
            "num_points": len(valid_src_pts),
            "accuracy": accuracy_report
        }
        
    except Exception as e:
        print(f"Error computing homography: {e}")
        with open(f"{debug_dir}/03_homography_error.txt", "w") as f:
            f.write(f"Homography computation failed: {e}\n")
            f.write(f"Valid points available: {len(valid_src_pts)}\n")
        
        return {"status": "error", "error": str(e), "num_points": len(valid_src_pts)}

def create_correspondence_visualization(frame, map_img, src_pts, dst_pts, labels, debug_dir):
    """
    Create side-by-side visualization showing point correspondences
    """
    # Resize images to same height for side-by-side display
    frame_h, frame_w = frame.shape[:2]
    map_h, map_w = map_img.shape[:2]
    
    target_height = 600
    frame_ratio = target_height / frame_h
    map_ratio = target_height / map_h
    
    frame_resized = cv2.resize(frame, (int(frame_w * frame_ratio), target_height))
    map_resized = cv2.resize(map_img, (int(map_w * map_ratio), target_height))
    
    # Create combined image
    combined_w = frame_resized.shape[1] + map_resized.shape[1]
    combined = np.zeros((target_height, combined_w, 3), dtype=np.uint8)
    
    # Place images side by side
    combined[:, :frame_resized.shape[1]] = frame_resized
    combined[:, frame_resized.shape[1]:] = map_resized
    
    # Draw correspondences
    offset_x = frame_resized.shape[1]
    
    for i, ((src_x, src_y), (dst_x, dst_y), label) in enumerate(zip(src_pts, dst_pts, labels)):
        # Scale coordinates
        scaled_src_x = int(src_x * frame_ratio)
        scaled_src_y = int(src_y * frame_ratio)
        scaled_dst_x = int(dst_x * map_ratio) + offset_x
        scaled_dst_y = int(dst_y * map_ratio)
        
        # Generate color for this correspondence
        color = (
            int(255 * (i / len(src_pts))),
            int(255 * (1 - i / len(src_pts))),
            128
        )
        
        # Draw points
        cv2.circle(combined, (scaled_src_x, scaled_src_y), 4, color, -1)
        cv2.circle(combined, (scaled_dst_x, scaled_dst_y), 4, color, -1)
        
        # Draw connection line
        cv2.line(combined, (scaled_src_x, scaled_src_y), (scaled_dst_x, scaled_dst_y), color, 1)
        
        # Add labels
        cv2.putText(combined, f"{i}:{label}", (scaled_src_x + 5, scaled_src_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv2.putText(combined, f"{i}:{label}", (scaled_dst_x + 5, scaled_dst_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Add labels
    cv2.putText(combined, "SOURCE (Frame)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "DESTINATION (Map)", (offset_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(f"{debug_dir}/03_point_correspondences.jpg", combined)
    
    return combined

def test_homography_accuracy(src_pts, dst_pts, labels, H, debug_dir):
    """
    Test homography accuracy by computing reprojection errors
    """
    src_array = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst_expected = np.array(dst_pts, dtype=np.float32)
    
    # Warp source points using homography
    dst_warped = cv2.perspectiveTransform(src_array, H).reshape(-1, 2)
    
    # Compute errors
    errors = np.sqrt(np.sum((dst_expected - dst_warped)**2, axis=1))
    
    # Create accuracy report
    with open(f"{debug_dir}/03_homography_accuracy.txt", "w") as f:
        f.write("HOMOGRAPHY ACCURACY REPORT\n")
        f.write("==========================\n\n")
        
        f.write(f"Total points used: {len(src_pts)}\n")
        f.write(f"Average error: {np.mean(errors):.2f} pixels\n")
        f.write(f"Maximum error: {np.max(errors):.2f} pixels\n")
        f.write(f"Minimum error: {np.min(errors):.2f} pixels\n")
        f.write(f"Standard deviation: {np.std(errors):.2f} pixels\n\n")
        
        f.write("POINT-BY-POINT ERRORS:\n")
        for i, (label, error, (sx, sy), (dx, dy), (wx, wy)) in enumerate(
            zip(labels, errors, src_pts, dst_pts, dst_warped)
        ):
            f.write(f"{i:2d}. {label:12s} | Error: {error:6.2f}px | "
                   f"Src: ({sx:6.1f},{sy:6.1f}) -> Expected: ({dx:6.1f},{dy:6.1f}) | "
                   f"Warped: ({wx:6.1f},{wy:6.1f})\n")
        
        # Identify problematic points
        high_error_threshold = np.mean(errors) + 2 * np.std(errors)
        high_error_points = [(labels[i], errors[i]) for i in range(len(errors)) if errors[i] > high_error_threshold]
        
        if high_error_points:
            f.write(f"\nHIGH ERROR POINTS (>{high_error_threshold:.1f}px):\n")
            for label, error in high_error_points:
                f.write(f"  {label}: {error:.2f}px\n")
    
    return {
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "num_points": len(src_pts)
    }

def create_warped_grid_visualization(frame, map_img, H, debug_dir):
    """
    Create visualization showing how a grid from the frame warps to the map
    """
    frame_h, frame_w = frame.shape[:2]
    
    # Create grid points
    grid_step = 50
    grid_pts = []
    for y in range(0, frame_h, grid_step):
        for x in range(0, frame_w, grid_step):
            grid_pts.append((x, y))
    
    # Warp grid points
    grid_array = np.array(grid_pts, dtype=np.float32).reshape(-1, 1, 2)
    warped_grid = cv2.perspectiveTransform(grid_array, H).reshape(-1, 2)
    
    # Create visualization on map
    map_vis = map_img.copy()
    map_h, map_w = map_img.shape[:2]
    
    # Draw warped grid points
    for (x, y) in warped_grid:
        if 0 <= x < map_w and 0 <= y < map_h:
            cv2.circle(map_vis, (int(x), int(y)), 2, (0, 255, 0), -1)
        else:
            # Draw out-of-bounds points at edge
            x_clipped = max(0, min(map_w-1, int(x)))
            y_clipped = max(0, min(map_h-1, int(y)))
            cv2.circle(map_vis, (x_clipped, y_clipped), 2, (0, 0, 255), -1)
    
    cv2.imwrite(f"{debug_dir}/04_warped_grid.jpg", map_vis)
    
    # Statistics
    in_bounds = sum(1 for (x, y) in warped_grid if 0 <= x < map_w and 0 <= y < map_h)
    total = len(warped_grid)
    
    with open(f"{debug_dir}/04_grid_warp_stats.txt", "w") as f:
        f.write("GRID WARP STATISTICS\n")
        f.write("====================\n\n")
        f.write(f"Total grid points: {total}\n")
        f.write(f"Points in bounds: {in_bounds}\n")
        f.write(f"Points out of bounds: {total - in_bounds}\n")
        f.write(f"Success rate: {in_bounds/total*100:.1f}%\n")

def create_2d_pitch_overlay(frame, map_img, H, debug_dir):
    """
    Create the 2D pitch overlay by warping the map onto the frame
    """
    try:
        frame_h, frame_w = frame.shape[:2]
        
        # Warp the map image to the frame perspective
        H_inv = np.linalg.inv(H)  # Inverse homography to go from map to frame
        warped_map = cv2.warpPerspective(map_img, H_inv, (frame_w, frame_h))
        
        # Create overlay by blending
        alpha = 0.4  # Transparency
        overlay = cv2.addWeighted(frame, 1-alpha, warped_map, alpha, 0)
        
        cv2.imwrite(f"{debug_dir}/05_2d_pitch_overlay.jpg", overlay)
        
        # Also save the warped map alone
        cv2.imwrite(f"{debug_dir}/05_warped_map_alone.jpg", warped_map)
        
        with open(f"{debug_dir}/05_overlay_info.txt", "w") as f:
            f.write("2D PITCH OVERLAY INFO\n")
            f.write("=====================\n\n")
            f.write(f"Frame size: {frame_w}x{frame_h}\n")
            f.write(f"Map warped to frame perspective\n")
            f.write(f"Overlay alpha: {alpha}\n")
            f.write(f"Files created:\n")
            f.write(f"  - 05_2d_pitch_overlay.jpg (blended result)\n")
            f.write(f"  - 05_warped_map_alone.jpg (warped map only)\n")
        
        return True
        
    except Exception as e:
        print(f"Error creating 2D overlay: {e}")
        with open(f"{debug_dir}/05_overlay_error.txt", "w") as f:
            f.write(f"2D overlay creation failed: {e}\n")
        return False

def run_full_diagnostic(frame, map_img_path, map_keypoints_path, debug_dir="enhanced_debug"):
    """
    Run the complete diagnostic suite
    """
    
    # Run comprehensive diagnostic
    result = create_comprehensive_diagnostic(frame, map_img_path, map_keypoints_path, debug_dir)
    
    # Create summary report
    with open(f"{debug_dir}/00_diagnostic_summary.txt", "w") as f:
        f.write("COMPREHENSIVE DIAGNOSTIC SUMMARY\n")
        f.write("================================\n\n")
        f.write(f"Map: {map_img_path}\n")
        f.write(f"Map keypoints: {map_keypoints_path}\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"Status: {result['status']}\n")
        f.write(f"Keypoints detected: {result['keypoints_detected']}\n")
        f.write(f"Areas detected: {result['areas_detected']}\n")
        f.write(f"Points matched: {result['matched_points']}\n")
        
        if result['status'] == 'success':
            accuracy = result['homography']['accuracy']
            f.write(f"Homography computed successfully\n")
            f.write(f"Average reprojection error: {accuracy['mean_error']:.2f} pixels\n")
            f.write(f"Maximum reprojection error: {accuracy['max_error']:.2f} pixels\n")
        
        f.write("\nFILES CREATED:\n")
        f.write("01_all_detections.jpg - All detected keypoints and areas\n")
        f.write("02_color_coded_keypoints.jpg - Green=matched, Red=unmatched\n")
        f.write("03_point_correspondences.jpg - Side-by-side point matching\n")
        f.write("04_warped_grid.jpg - Grid transformation visualization\n")
        f.write("05_2d_pitch_overlay.jpg - Final 2D pitch overlay\n")
        f.write("\nCheck the accompanying .txt files for detailed analysis.\n")
    
    print(f"\nDiagnostic complete! Check {debug_dir} for results.")
    return result

if __name__ == "__main__":
    # Example usage
    frame_path = "path/to/your/frame.jpg"
    map_img_path = "calib/map.jpg"
    map_keypoints_path = "calib/map_keypoints.json"
    
    result = run_full_diagnostic(frame_path, map_img_path, map_keypoints_path)