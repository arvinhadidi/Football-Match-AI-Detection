import json
import os
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from tkinter import Tk, filedialog
import cv2
from player_ball_assigner import PlayerBallAssigner
from utils.bbox_utils import get_center_of_bbox
from homography.areas_keypoints_homography import match_keypoints_to_areas, compute_homography, warp_point
from homography.enhanced_homography_diagnostic import run_full_diagnostic

with open("calib/map_keypoints.json") as f:
    map_keypoints = json.load(f)
 
with open("calib/shortened_map_keypoints.json") as f:
    shortened_map_keypoints = json.load(f)

def load_map_keypoints(json_path="calib/map_keypoints.json"):
    """Returns a dict mapping label→[x,y] in map-image coordinates."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def create_2d_pitch_overlay(frame, map_img, H, alpha=0.4):
    """
    Create the 2D pitch overlay by warping the map onto the frame
    Returns the blended overlay image, or original frame if H is None
    """
    try:
        if H is None:
            return frame
            
        frame_h, frame_w = frame.shape[:2]
        
        # Warp the map image to the frame perspective
        H_inv = np.linalg.inv(H)  # Inverse homography to go from map to frame
        warped_map = cv2.warpPerspective(map_img, H_inv, (frame_w, frame_h))
        
        # Create overlay by blending
        overlay = cv2.addWeighted(frame, 1-alpha, warped_map, alpha, 0)
        
        return overlay
        
    except Exception as e:
        print(f"Error creating 2D overlay: {e}")
        return frame  # Return original frame if overlay fails

def safe_compute_homography(src_pts, dst_pts):
    """
    Wrapper around compute_homography that returns None instead of raising exceptions
    """
    try:
        return compute_homography(src_pts, dst_pts)
    except Exception as e:
        print(f"Homography computation failed: {e}")
        return None

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

    # Run diagnostic on first frame for initial setup
    result = run_full_diagnostic(
        frame=video_frames[0],
        map_img_path="calib/map.jpg", 
        map_keypoints_path="calib/map_keypoints.json"
    )

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
    shortened_map_keypoints = load_map_keypoints("calib/map_keypoints.json")

    # 9) Homography + warp loop
    RECALIBRATE_EVERY  = 10
    H                  = None  # Start as None instead of identity
    warped_player_info = []   # per-frame list of (pid, (x,y))
    warped_ball_info   = []   # per-frame (x,y)
    homography_list    = []   # Store homography for each frame
    successful_homographies = 0  # Track how many successful homographies we compute

    fh, fw = video_frames[0].shape[:2]

    for fn, frame in enumerate(video_frames):
        # — 9A) (Re)compute H every RECALIBRATE_EVERY frames —
        if fn % RECALIBRATE_EVERY == 0:
            image_pts, labels = match_keypoints_to_areas(frame)
            print(f"[Frame {fn}] IMAGEPTS: {image_pts}")
            print(f"[Frame {fn}] LABELS: {labels}")
            
            # filter out any that lie on the very edge or low-confidence
            filtered = []
            for (x,y), lbl in zip(image_pts, labels):
                if 5 < x < fw-5 and 5 < y < fh-5 and lbl in shortened_map_keypoints:
                    filtered.append((x,y,lbl))

            print(f"[Frame {fn}] FILTERED: {filtered}")

            if len(filtered) >= 4:
                src_pts = [(x,y) for x,y,_ in filtered]
                dst_pts = [shortened_map_keypoints[lbl] for _,_,lbl in filtered]
                
                # Use safe wrapper that returns None on failure
                best_H = safe_compute_homography(src_pts, dst_pts)
                
                if best_H is not None:
                    H = best_H
                    successful_homographies += 1
                    print(f"[Frame {fn}] Successfully computed homography #{successful_homographies}")
                else:
                    print(f"[Frame {fn}] Failed to compute homography - will skip pitch mapping")
                    # H remains None or keeps previous value
                    
                print(f"[Frame {fn}] Attempted homography with {len(filtered)} corners")
            else:
                print(f"[Frame {fn}] Only {len(filtered)} usable corners; skipping homography computation")

        # Store the current homography for this frame (could be None)
        homography_list.append(H.copy() if H is not None else None)

        # — 9B) Warp each player center using current H (only if H is valid) —
        frame_players = []
        if H is not None:
            for pid, det in tracks["players"][fn].items():
                try:
                    cx, cy = get_center_of_bbox(det["bbox"])
                    wx, wy = warp_point((cx, cy), H)
                    frame_players.append((pid, (wx, wy)))
                except Exception as e:
                    print(f"[Frame {fn}] Failed to warp player {pid}: {e}")
                    frame_players.append((pid, (np.nan, np.nan)))
        else:
            # No valid homography - add NaN positions for all players
            for pid, det in tracks["players"][fn].items():
                frame_players.append((pid, (np.nan, np.nan)))

        # — 9C) Warp ball center if present and H is valid —
        if H is not None:
            bb = tracks["ball"][fn].get(1, {}).get("bbox")
            if bb:
                try:
                    bx, by = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
                    ball_pt = warp_point((bx, by), H)
                except Exception as e:
                    print(f"[Frame {fn}] Failed to warp ball: {e}")
                    ball_pt = (np.nan, np.nan)
            else:
                ball_pt = (np.nan, np.nan)
        else:
            ball_pt = (np.nan, np.nan)

        # 9D) Always append (even if empty/NaN)
        warped_player_info.append(frame_players)
        warped_ball_info.append(ball_pt)

    print(f"\nHomography computation summary:")
    print(f"Total frames processed: {len(video_frames)}")
    print(f"Successful homographies: {successful_homographies}")
    print(f"Homography attempts: {len(video_frames) // RECALIBRATE_EVERY + (1 if len(video_frames) % RECALIBRATE_EVERY > 0 else 0)}")

    # 10) Build three types of frames: annotated, pure-pitch, and 2D overlay
    annotated_frames = []
    pitch_frames     = []
    overlay_frames   = []

    for fn, frame in enumerate(video_frames):
        # --- A) On-field annotation (always works) ---
        annotated = tracker.draw_annotations(
            [frame],
            {"players":[tracks["players"][fn]],
             "referees":[tracks["referees"][fn]],
             "ball":[tracks["ball"][fn]]}
        )[0]

        annotated_frames.append(annotated)

        # --- B) Pure pitch + dots (only if we have valid warped positions) ---
        pitch_full = map_img.copy()
        has_valid_positions = False
        
        for pid, (wx, wy) in warped_player_info[fn]:
            if not (np.isnan(wx) or np.isnan(wy)):
                has_valid_positions = True
                color = tracks["players"][fn][pid]["team_color"]
                try:
                    color = tuple(int(c) for c in color)
                except Exception:
                    color = (255, 255, 255)
                cv2.circle(pitch_full, (int(wx), int(wy)), 5, color, -1)

        # Add ball if valid
        bx, by = warped_ball_info[fn]
        if not (np.isnan(bx) or np.isnan(by)):
            has_valid_positions = True
            cv2.circle(pitch_full, (int(bx), int(by)), 5, (0,255,0), -1)

        # If no valid positions, add text overlay indicating no homography
        if not has_valid_positions:
            cv2.putText(pitch_full, "No valid homography for this frame", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        pitch_frames.append(pitch_full)

        # --- C) 2D Pitch Overlay (only if homography is valid) ---
        current_H = homography_list[fn]
        if current_H is not None:
            overlay_frame = create_2d_pitch_overlay(frame, map_img, current_H, alpha=0.4)
            
            # Add player tracking dots on the overlay
            for pid, det in tracks["players"][fn].items():
                if det.get('bbox'):
                    cx, cy = get_center_of_bbox(det["bbox"])
                    color = det.get('team_color', (255, 255, 255))
                    try:
                        color = tuple(int(c) for c in color)
                    except Exception:
                        color = (255, 255, 255)
                    cv2.circle(overlay_frame, (int(cx), int(cy)), 4, color, -1)
            
            # Add ball dot if present
            bb = tracks["ball"][fn].get(1, {}).get("bbox")
            if bb:
                bx, by = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
                cv2.circle(overlay_frame, (int(bx), int(by)), 4, (0, 255, 0), -1)
        else:
            # No valid homography - use original frame with text overlay
            overlay_frame = frame.copy()
            cv2.putText(overlay_frame, "No pitch overlay available", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        overlay_frames.append(overlay_frame)

    # 11) Save all three videos
    save_video(annotated_frames, output_path, fps=20.0)
    pitch_out = f"output_videos/{base}_pitch.avi"
    save_video(pitch_frames, pitch_out, fps=20.0)
    overlay_out = f"output_videos/{base}_overlay.avi"
    save_video(overlay_frames, overlay_out, fps=20.0)
    
    print(f"\nSaved annotated video   → {output_path}")
    print(f"Saved pitch-only video  → {pitch_out}")
    print(f"Saved 2D overlay video  → {overlay_out}")


if __name__ == "__main__":
    main()