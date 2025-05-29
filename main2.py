import json
import os
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from tkinter import Tk, filedialog
import cv2
from player_ball_assigner import PlayerBallAssigner
from homography.homography import detect_pitch_keypoints, compute_homography, warp_point, return_api_request
from utils.bbox_utils import get_center_of_bbox
from homography.homography_diagnostic import diagnose_homography, run_diagnostics
from homography.fix_homography_issues import fix_homography_issues
from homography.hough import detect_pitch_keypoints_with_fallback

# Add the new robust method import
from homography.hough import detect_pitch_keypoints_hough_robust

with open("calib/map_keypoints.json") as f:
    map_keypoints = json.load(f)

with open("calib/shortened_map_keypoints.json") as f:
    shortened_map_keypoints = json.load(f)

def load_map_keypoints(json_path="calib/map_keypoints.json"):
    """Returns a dict mapping label→[x,y] in map-image coordinates."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def detect_keypoints_with_multiple_fallbacks(frame, debug=False):
    """
    Multi-method keypoint detection with robust fallbacks
    """
    methods = [
        ("Robust Hough", lambda f: detect_pitch_keypoints_hough_robust(f, debug=debug)),
        ("Original Fallback", lambda f: detect_pitch_keypoints_with_fallback(f)),
        ("Standard Detection", lambda f: detect_pitch_keypoints(f))
    ]
    
    for method_name, method_func in methods:
        try:
            if debug:
                print(f"Trying {method_name}...")
            
            image_pts, labels = method_func(frame)
            
            if len(image_pts) >= 3:  # Success criteria
                if debug:
                    print(f"{method_name} succeeded with {len(image_pts)} keypoints")
                return image_pts, labels
            elif debug:
                print(f"{method_name} found only {len(image_pts)} keypoints")
                
        except Exception as e:
            if debug:
                print(f"{method_name} failed with error: {e}")
            continue
    
    if debug:
        print("All keypoint detection methods failed")
    return [], []

def filter_keypoints_adaptive(image_pts, labels, frame_shape, map_keypoints_available):
    """
    More adaptive keypoint filtering that doesn't rely on strict expected positions
    """
    if not image_pts:
        return []
    
    fh, fw = frame_shape[:2]
    filtered = []
    
    # Group points by label to avoid duplicates
    label_groups = {}
    for (x, y), lbl in zip(image_pts, labels):
        if lbl not in label_groups:
            label_groups[lbl] = []
        label_groups[lbl].append((x, y))
    
    # For each label, keep the best point (if label exists in map)
    for lbl, points in label_groups.items():
        if lbl not in map_keypoints_available:
            continue
            
        # Filter points that are too close to edges
        valid_points = []
        for x, y in points:
            if 10 < x < fw-10 and 10 < y < fh-10:  # Reasonable margin
                valid_points.append((x, y))
        
        if not valid_points:
            continue
            
        if len(valid_points) == 1:
            x, y = valid_points[0]
            filtered.append((x, y, lbl))
        else:
            # Multiple points with same label - choose one near center of group
            center_x = sum(p[0] for p in valid_points) / len(valid_points)
            center_y = sum(p[1] for p in valid_points) / len(valid_points)
            
            # Find point closest to group center
            best_point = min(valid_points, 
                           key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
            
            filtered.append((best_point[0], best_point[1], lbl))
    
    return filtered

def main():
    root = Tk()
    root.withdraw()  # hide the empty tkinter window
    input_path = filedialog.askopenfilename(
        title="Select input video",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    root.destroy()
    if not input_path:
        print("No file selected. Exiting.")
        return

    # Compute output filename (same base + "_output.avi")
    base, ext = os.path.splitext(os.path.basename(input_path))
    output_path = (f"output_videos/{base}_output.avi")
    # read video
    video_frames = read_video(input_path, target_fps=20.0)

    # -----------------------------------------------------------------------------

    #initialise tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path = "stubs/track_stubs.pkl")

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    first_frame_players = {
        pid: det
        for pid, det in tracks['players'][0].items()
        if pid not in tracks['referees'][0]
    }

    # Use tracker.team_assigner instead of team_assigner
    tracker.team_assigner.assign_team_color(video_frames[0], first_frame_players)
    debug_line_detection(video_frames[0])

    # And in your player assignment loop:
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = tracker.team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = tracker.team_assigner.team_colors[team]

    # Assign ball to closest player
    player_assigner = PlayerBallAssigner()
    team_ball_control= []

    for frame_num, player_track in enumerate(tracks['players']):
        # get the smoothed/interpolated ball bbox
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            # mark who has the ball in our tracks dict
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team']
            )
        else:
            # if nobody new, assume same team as last frame
            team_ball_control.append(team_ball_control[-1] if team_ball_control else None)

    team_ball_control = np.array(team_ball_control)

    map_img = cv2.imread("calib/map.jpg")
    if map_img is None:
        raise FileNotFoundError("calib/map.jpg not found!")
    shortened_map_keypoints = load_map_keypoints("calib/shortened_map_keypoints.json")

    # 9) Homography + warp loop
    RECALIBRATE_EVERY  = 30
    H                  = np.eye(3, dtype=np.float32)  # start as identity
    warped_player_info = []   # per-frame list of (pid, (x,y))
    warped_ball_info   = []   # per-frame (x,y)

    fh, fw = video_frames[0].shape[:2]

    # Track homography quality for adaptive recalibration
    last_successful_frame = -1
    consecutive_failures = 0
    
    for fn, frame in enumerate(video_frames):
        # — 9A) Adaptive homography recalibration —
        should_recalibrate = (
            fn % RECALIBRATE_EVERY == 0 or  # Regular interval
            consecutive_failures >= 3 or     # Too many recent failures
            fn - last_successful_frame > RECALIBRATE_EVERY * 2  # Been too long since success
        )
        
        if should_recalibrate:
            print(f"\n[Frame {fn}] Attempting homography recalibration...")
            
            # Use the robust multi-method detection
            image_pts, labels = detect_keypoints_with_multiple_fallbacks(frame, debug=True)
            
            print(f"Detected keypoints: {len(image_pts)}")
            if image_pts:
                print(f"Labels: {labels}")
            
            # Use adaptive filtering
            filtered = filter_keypoints_adaptive(image_pts, labels, frame.shape, shortened_map_keypoints)
            
            print(f"Filtered keypoints: {len(filtered)}")
            if filtered:
                print(f"Filtered labels: {[f[2] for f in filtered]}")

            if len(filtered) >= 4:
                src_pts = [(x, y) for x, y, _ in filtered]
                dst_pts = [shortened_map_keypoints[lbl] for _, _, lbl in filtered]
                
                # Visualize detected keypoints
                for x, y, lbl in filtered:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(frame, lbl, (int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                best_H = compute_homography(src_pts, dst_pts)
                
                if best_H is not None:
                    H = best_H
                    last_successful_frame = fn
                    consecutive_failures = 0
                    print(f"[Frame {fn}] Successfully computed homography with {len(filtered)} keypoints")
                else:
                    consecutive_failures += 1
                    print(f"[Frame {fn}] Homography computation failed")
            else:
                consecutive_failures += 1
                print(f"[Frame {fn}] Only {len(filtered)} usable keypoints; need at least 4")

        # — 9B) Warp each player center using current H —
        frame_players = []
        for pid, det in tracks["players"][fn].items():
            cx, cy = get_center_of_bbox(det["bbox"])
            wx, wy = warp_point((cx, cy), H)
            frame_players.append((pid, (wx, wy)))

        # — 9C) Warp ball center if present —
        bb = tracks["ball"][fn].get(1, {}).get("bbox")
        if bb:
            bx, by = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
            ball_pt = warp_point((bx, by), H)
        else:
            ball_pt = (np.nan, np.nan)

        # 9D) Always append (even if empty)
        warped_player_info.append(frame_players)
        warped_ball_info.append(ball_pt)

    # 10) Build both annotated frames + pure-pitch frames
    annotated_frames = []
    pitch_frames     = []

    for fn, frame in enumerate(video_frames):
        # --- A) On-field annotation + mini-map (exactly as before) ---
        annotated = tracker.draw_annotations(
            [frame],
            {"players":[tracks["players"][fn]],
             "referees":[tracks["referees"][fn]],
             "ball":[tracks["ball"][fn]]}
        )[0]

        annotated_frames.append(annotated)

        # --- B) Pure pitch + dots, full-frame size ---
        pitch_full = map_img.copy()
        for pid, (wx, wy) in warped_player_info[fn]:
            if not (np.isnan(wx) or np.isnan(wy)):
                color = tracks["players"][fn][pid]["team_color"]

                # ensure it's a proper BGR tuple of ints
                try:
                    color = tuple(int(c) for c in color)
                except Exception:
                    # fallback white if something unexpected
                    color = (255, 255, 255)

                cv2.circle(pitch_full, (int(wx), int(wy)), 5, color, -1)

        # if you also want the ball:
        bx, by = warped_ball_info[fn]
        if not (np.isnan(bx) or np.isnan(by)):
            cv2.circle(pitch_full, (int(bx), int(by)), 5, (0,255,0), -1)

        pitch_frames.append(pitch_full)

    # 11) Save both videos
    save_video(annotated_frames, output_path, fps=20.0)
    pitch_out = f"output_videos/{base}_pitch.avi"
    save_video(pitch_frames, pitch_out, fps=20.0)
    print(f"Saved annotated video   → {output_path}")
    print(f"Saved pitch-only video  → {pitch_out}")


if __name__ == "__main__":
    main()