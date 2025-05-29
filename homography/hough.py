import cv2
import numpy as np
from collections import defaultdict
import math
from homography.homography import detect_pitch_keypoints

def detect_pitch_keypoints_with_fallback(frame):
    """
    Try Hough method first, fallback to original if insufficient keypoints
    """
    # Try Hough method
    image_pts, labels = detect_pitch_keypoints_hough(frame)
    
    if len(image_pts) >= 4:
        print(f"Hough method found {len(image_pts)} keypoints")
        return image_pts, labels
    else:
        print(f"Hough method only found {len(image_pts)} keypoints, falling back to original method")
        # Fallback to your original detect_pitch_keypoints function
        return detect_pitch_keypoints(frame)

def detect_pitch_keypoints_hough(frame):
    """
    Uses Hough Line Transform to detect field lines, then finds intersections
    to generate the most vital keypoints for robust homography.
    Returns two parallel lists: (image_pts, labels).
    """
    # 1) Detect and filter lines
    lines = detect_field_lines_enhanced(frame)
    if len(lines) < 4:
        return [], []
    
    # 2) Group similar lines to reduce noise
    grouped_lines = group_similar_lines(lines)
    
    # 3) Classify lines by orientation
    classified_lines = classify_lines_improved(grouped_lines, frame.shape)
    
    # 4) Find intersections between different line types
    intersections = find_key_intersections(classified_lines, frame)
    
    # 5) Map to keypoint labels with strict filtering
    image_pts, labels = map_intersections_to_keypoints_improved(intersections, frame.shape)
    
    # 6) Remove duplicates and filter
    filtered_pts, filtered_labels = filter_and_deduplicate_keypoints(image_pts, labels, frame.shape)
    
    return filtered_pts, filtered_labels

def detect_field_lines_enhanced(frame):
    """
    Enhanced line detection with better preprocessing for football pitches
    """
    # Convert to different color spaces for better line detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for white lines on grass
    # White lines typically have low saturation and high value
    white_mask1 = cv2.inRange(gray, 160, 255)
    
    # HSV mask for white/light colors 
    white_lower = np.array([0, 0, 160])
    white_upper = np.array([180, 60, 255])
    white_mask2 = cv2.inRange(hsv, white_lower, white_upper)
    
    # Combine masks
    white_mask = cv2.bitwise_or(white_mask1, white_mask2)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    # white_mask = cv2.dilate(white_mask, kernel, iterations=1)
    
    # Edge detection with adaptive thresholds
    edges = cv2.Canny(white_mask, 20, 80, apertureSize=3)
    
    # Use probabilistic Hough transform for better line detection
    line_segments = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=40, 
                                    minLineLength=30, 
                                    maxLineGap=30)
    
    lines = []
    if line_segments is not None:
        for segment in line_segments:
            x1, y1, x2, y2 = segment[0]
            
            # Convert to rho-theta format
            if x2 == x1:  # Vertical line
                theta = np.pi / 2
                rho = abs(x1)
            else:
                # Calculate angle and distance
                dx, dy = x2 - x1, y2 - y1
                theta = math.atan2(dy, dx)
                
                # Normalize theta to [0, pi)
                if theta < 0:
                    theta += np.pi
                    
                # Calculate perpendicular distance from origin
                rho = abs(x1 * math.sin(theta) - y1 * math.cos(theta))
            
            lines.append((rho, theta))
    
    return lines

def group_similar_lines(lines, rho_threshold=10, theta_threshold=0.1):
    """
    Group similar lines to reduce duplicates
    """
    if not lines:
        return []
    
    grouped = []
    used = [False] * len(lines)
    
    for i, (rho1, theta1) in enumerate(lines):
        if used[i]:
            continue
            
        # Start a new group
        group_rho = [rho1]
        group_theta = [theta1]
        used[i] = True
        
        # Find similar lines
        for j, (rho2, theta2) in enumerate(lines):
            if used[j]:
                continue
                
            # Check if lines are similar
            rho_diff = abs(rho1 - rho2)
            theta_diff = min(abs(theta1 - theta2), np.pi - abs(theta1 - theta2))
            
            if rho_diff < rho_threshold and theta_diff < theta_threshold:
                group_rho.append(rho2)
                group_theta.append(theta2)
                used[j] = True
        
        # Average the group to get representative line
        avg_rho = np.mean(group_rho)
        avg_theta = np.mean(group_theta)
        grouped.append((avg_rho, avg_theta))
    
    return grouped

def classify_lines_improved(lines, frame_shape):
    """
    Improved line classification with stricter angle criteria
    """
    h, w = frame_shape[:2]
    classified = {
        'horizontal': [],
        'vertical': [],
        'center_vertical': []
    }
    
    for rho, theta in lines:
        angle_deg = math.degrees(theta)
        
        # Normalize angle to 0-180 range
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
            
        # Stricter classification
        if angle_deg < 15:  # Horizontal lines (goal lines, penalty box lines)
            classified['horizontal'].append((rho, theta))
        elif angle_deg > 75:  # Vertical lines (sidelines, penalty box sides)
            # Check if it's the center line
            if theta < np.pi/2:
                x_pos = rho / math.cos(theta)
            else:
                x_pos = rho / math.cos(np.pi - theta)
                
            # Center line detection (within 15% of frame center)
            if abs(x_pos - w/2) < w * 0.15:
                classified['center_vertical'].append((rho, theta))
            else:
                classified['vertical'].append((rho, theta))
    
    return classified

def find_key_intersections(classified_lines, frame):
    """
    Find key intersections with better filtering
    """
    intersections = []
    h, w = frame.shape[:2]
    
    # Horizontal-Vertical intersections (corners, penalty box corners)
    for h_line in classified_lines['horizontal']:
        for v_line in classified_lines['vertical']:
            intersection = line_intersection(h_line, v_line)
            if intersection and is_valid_intersection(intersection, (w, h)):
                intersections.append({
                    'point': intersection,
                    'type': 'corner',
                    'lines': [h_line, v_line]
                })
    
    # Horizontal-Center intersections (center circle intersections)
    for h_line in classified_lines['horizontal']:
        for cv_line in classified_lines['center_vertical']:
            intersection = line_intersection(h_line, cv_line)
            if intersection and is_valid_intersection(intersection, (w, h)):
                intersections.append({
                    'point': intersection,
                    'type': 'center',
                    'lines': [h_line, cv_line]
                })
    
    return intersections

def is_valid_intersection(point, frame_size):
    """
    Check if intersection point is valid (within frame bounds with margin)
    """
    x, y = point
    w, h = frame_size
    margin = 10
    
    return (margin < x < w - margin and 
            margin < y < h - margin)

def line_intersection(line1, line2):
    """
    Find intersection point of two lines in rho-theta format
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    
    # Check if lines are too similar (parallel)
    angle_diff = abs(theta1 - theta2)
    if angle_diff < 0.1 or abs(angle_diff - np.pi) < 0.1:
        return None
    
    # Convert to cartesian form: ax + by = c
    a1, b1 = math.cos(theta1), math.sin(theta1)
    a2, b2 = math.cos(theta2), math.sin(theta2)
    c1, c2 = rho1, rho2
    
    # Solve system of equations
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    
    return (int(round(x)), int(round(y)))

def map_intersections_to_keypoints_improved(intersections, frame_shape):
    """
    Map intersections to keypoint labels with strict position criteria
    """
    h, w = frame_shape[:2]
    image_pts = []
    labels = []
    
    # Group intersections by region to avoid duplicates
    regions = defaultdict(list)
    
    for intersection in intersections:
        x, y = intersection['point']
        intersection_type = intersection['type']
        
        # Determine region
        region_x = 'left' if x < w/3 else 'right' if x > 2*w/3 else 'center'
        region_y = 'top' if y < h/3 else 'bottom' if y > 2*h/3 else 'middle'
        region_key = f"{region_y}_{region_x}"
        
        regions[region_key].append((x, y, intersection_type))
    
    # Process each region and select best representative point
    for region_key, points in regions.items():
        if not points:
            continue
            
        # Get the most central point in this region
        if len(points) == 1:
            x, y, int_type = points[0]
        else:
            # Find centroid of points in this region
            xs, ys = zip(*[(p[0], p[1]) for p in points])
            x, y = int(np.mean(xs)), int(np.mean(ys))
            int_type = points[0][2]  # Use type from first point
        
        # Determine label based on region and type
        label = determine_label_by_region(region_key, int_type, x, y, frame_shape)
        
        if label:
            image_pts.append((x, y))
            labels.append(label)
    
    return image_pts, labels

def determine_label_by_region(region_key, intersection_type, x, y, frame_shape):
    """
    Determine keypoint label based on region with strict mapping
    """
    h, w = frame_shape[:2]
    norm_x, norm_y = x / w, y / h
    
    # Strict region-based labeling
    region_mapping = {
        'top_left': ['TLC', 'TL18MC'],
        'top_right': ['TRC', 'TR18MC'], 
        'top_center': ['TCS'],
        'bottom_left': ['BLC', 'BL18MC'],
        'bottom_right': ['BRC', 'BR18MC'],
        'bottom_center': ['BCS'],
        'middle_left': ['LML'],
        'middle_right': ['RML']
    }
    
    possible_labels = region_mapping.get(region_key, [])
    
    if not possible_labels:
        return None
    
    # Select most appropriate label based on exact position
    if intersection_type == 'corner':
        # For corner intersections, prefer main corners if near edges
        if norm_x < 0.2 and norm_y < 0.2:
            return 'TLC'
        elif norm_x > 0.8 and norm_y < 0.2:
            return 'TRC'
        elif norm_x < 0.2 and norm_y > 0.8:
            return 'BLC'
        elif norm_x > 0.8 and norm_y > 0.8:
            return 'BRC'
        # Otherwise, penalty box corners
        elif 'TL18MC' in possible_labels and 0.15 < norm_x < 0.45 and 0.15 < norm_y < 0.4:
            return 'TL18MC'
        elif 'TR18MC' in possible_labels and 0.55 < norm_x < 0.85 and 0.15 < norm_y < 0.4:
            return 'TR18MC'
        elif 'BL18MC' in possible_labels and 0.15 < norm_x < 0.45 and 0.6 < norm_y < 0.85:
            return 'BL18MC'
        elif 'BR18MC' in possible_labels and 0.55 < norm_x < 0.85 and 0.6 < norm_y < 0.85:
            return 'BR18MC'
        # Middle sideline points
        elif 'LML' in possible_labels and norm_x < 0.2 and 0.4 < norm_y < 0.6:
            return 'LML'
        elif 'RML' in possible_labels and norm_x > 0.8 and 0.4 < norm_y < 0.6:
            return 'RML'
            
    elif intersection_type == 'center':
        # Center circle intersections
        if 0.45 < norm_x < 0.55:
            if norm_y < 0.45:
                return 'TCS'
            elif norm_y > 0.55:
                return 'BCS'
    
    return None

def filter_and_deduplicate_keypoints(image_pts, labels, frame_shape, min_distance=30):
    """
    Remove duplicate keypoints that are too close to each other
    """
    if not image_pts:
        return [], []
    
    h, w = frame_shape[:2]
    filtered_pts = []
    filtered_labels = []
    
    # Group points by label
    label_groups = defaultdict(list)
    for i, (pt, label) in enumerate(zip(image_pts, labels)):
        label_groups[label].append((i, pt))
    
    # For each label, keep only the best representative point
    for label, points in label_groups.items():
        if len(points) == 1:
            idx, pt = points[0]
            filtered_pts.append(pt)
            filtered_labels.append(label)
        else:
            # Multiple points with same label - keep the most central one
            # or the one closest to expected position
            expected_pos = get_expected_position(label, frame_shape)
            
            if expected_pos:
                # Keep point closest to expected position
                best_idx = min(points, key=lambda x: 
                    math.sqrt((x[1][0] - expected_pos[0])**2 + (x[1][1] - expected_pos[1])**2))
                filtered_pts.append(best_idx[1])
                filtered_labels.append(label)
            else:
                # Keep first point if no expected position
                filtered_pts.append(points[0][1])
                filtered_labels.append(label)
    
    # Final edge filtering
    final_pts = []
    final_labels = []
    
    for (x, y), label in zip(filtered_pts, filtered_labels):
        if 5 < x < w-5 and 5 < y < h-5:
            final_pts.append((x, y))
            final_labels.append(label)
    
    return final_pts, final_labels

def get_expected_position(label, frame_shape):
    """
    Get expected relative position for a keypoint label
    """
    h, w = frame_shape[:2]
    
    # Expected relative positions (normalized coordinates)
    expected_positions = {
        'TLC': (0.05, 0.05),
        'TRC': (0.95, 0.05),
        'BLC': (0.05, 0.95),
        'BRC': (0.95, 0.95),
        'TL18MC': (0.25, 0.25),
        'TR18MC': (0.75, 0.25),
        'BL18MC': (0.25, 0.75),
        'BR18MC': (0.75, 0.75),
        'LML': (0.05, 0.5),
        'RML': (0.95, 0.5),
        'TCS': (0.5, 0.4),
        'BCS': (0.5, 0.6)
    }
    
    if label in expected_positions:
        norm_x, norm_y = expected_positions[label]
        return (int(norm_x * w), int(norm_y * h))
    
    return None

def debug_line_detection(frame, save_debug=True):
    """
    Debug function to visualize line detection process
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Show masks
    white_mask1 = cv2.inRange(gray, 160, 255)
    white_lower = np.array([0, 0, 160])
    white_upper = np.array([180, 60, 255])
    white_mask2 = cv2.inRange(hsv, white_lower, white_upper)
    white_mask = cv2.bitwise_or(white_mask1, white_mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    edges = cv2.Canny(white_mask, 20, 80, apertureSize=3)
    
    if save_debug:
        cv2.imwrite('debug_white_mask.jpg', white_mask)
        cv2.imwrite('debug_edges.jpg', edges)
        print("Debug images saved: debug_white_mask.jpg, debug_edges.jpg")
    
    # Detect lines
    line_segments = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=40, minLineLength=30, maxLineGap=30)
    
    print(f"Found {len(line_segments) if line_segments is not None else 0} line segments")
    
    # Draw lines on original frame
    debug_frame = frame.copy()
    if line_segments is not None:
        for segment in line_segments:
            x1, y1, x2, y2 = segment[0]
            cv2.line(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if save_debug:
        cv2.imwrite('debug_lines.jpg', debug_frame)
        print("Lines visualization saved: debug_lines.jpg")
    
    return line_segments