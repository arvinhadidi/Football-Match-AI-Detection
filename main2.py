import json
import os
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from tkinter import Tk, filedialog
import cv2
from player_ball_assigner import PlayerBallAssigner
# from homography.homography import detect_pitch_keypoints, compute_homography, warp_point, return_api_request
# from homography.homography_areas import detect_pitch_keypoints, compute_homography, warp_point, return_api_request
from utils.bbox_utils import get_center_of_bbox
from homography.homography_diagnostic import diagnose_homography, run_diagnostics
from homography.fix_homography_issues import fix_homography_issues
from homography.areas_keypoints_homography import match_keypoints_to_areas, compute_homography, warp_point
# from homography.hough import debug_line_detection, detect_pitch_keypoints_with_fallback
# from homography.hough import detect_pitch_keypoints_hough, debug_line_detection
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
    Returns the blended overlay image
    """
    try:
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
    H                  = np.eye(3, dtype=np.float32)  # start as identity
    warped_player_info = []   # per-frame list of (pid, (x,y))
    warped_ball_info   = []   # per-frame (x,y)
    homography_list    = []   # Store homography for each frame

    fh, fw = video_frames[0].shape[:2]

    for fn, frame in enumerate(video_frames):
        # — 9A) (Re)compute H every RECALIBRATE_EVERY frames —
        if fn % RECALIBRATE_EVERY == 0: # and fn != 0
            # image_pts, labels = detect_pitch_keypoints(frame)
            image_pts, labels = match_keypoints_to_areas(frame)
            print("IMAGEPTS")
            print(image_pts)
            print("LABELS")
            print(labels)
            # filter out any that lie on the very edge or low-confidence
            filtered = []
            for (x,y), lbl in zip(image_pts, labels):
                if 5 < x < fw-5 and 5 < y < fh-5 and lbl in shortened_map_keypoints:
                    filtered.append((x,y,lbl))

            print("FILTERED")
            print(filtered)

            if len(filtered) >= 4:
                src_pts = [(x,y) for x,y,_ in filtered]
                dst_pts = [shortened_map_keypoints[lbl] for _,_,lbl in filtered]
                cv2.circle(frame,(int(x),int(y)),5,(0,0,255),-1)
                best_H = compute_homography(src_pts, dst_pts)
                # best_H, best_score = fix_homography_issues(frame, tracks, map_img, map_keypoints)
                if best_H is not None:
                    H = best_H
                    print(f"[Frame {fn}] Using optimized homography")
                else:
                    print(f"[Frame {fn}] Could not compute reliable homography")

                print(f"[Frame {fn}] recomputed H with {len(filtered)} corners")
            else:
                print(f"[Frame {fn}] only {len(filtered)} usable corners; reusing last H")

        # Store the current homography for this frame
        homography_list.append(H.copy())

        # — 9B) Warp each player center using current H —
        frame_players = []
        for pid, det in tracks["players"][fn].items():
            cx, cy = get_center_of_bbox(det["bbox"])
            wx, wy = warp_point((cx, cy), H)
            frame_players.append((pid, (wx, wy)))
            # print((pid, (wx, wy)))

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


    # 10) Build three types of frames: annotated, pure-pitch, and 2D overlay
    annotated_frames = []
    pitch_frames     = []
    overlay_frames   = []  # New: 2D pitch overlay frames

    for fn, frame in enumerate(video_frames):
        # --- A) On-field annotation + mini-map (exactly as before) ---
        annotated = tracker.draw_annotations(
            [frame],
            {"players":[tracks["players"][fn]],
             "referees":[tracks["referees"][fn]],
             "ball":[tracks["ball"][fn]]}
        )[0]

        # # build thumbnail overlay
        # pitch_map_thumb = map_img.copy()
        # for pid, (wx, wy) in warped_player_info[fn]:
        #     if not (np.isnan(wx) or np.isnan(wy)):
        #         color = tracks["players"][fn][pid]["team_color"]
        #         cv2.circle(pitch_map_thumb, (int(wx), int(wy)), 5, color, -1)

        # fh, fw = annotated.shape[:2]
        # mh, mw = pitch_map_thumb.shape[:2]
        # tw = min(int(fw * 0.4), fw-20)
        # th = min(int(tw * (mh/mw)), fh-20)
        # thumb = cv2.resize(pitch_map_thumb, (tw, th))
        # annotated[10:10+th, 10:10+tw] = thumb

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

        # --- C) NEW: 2D Pitch Overlay ---
        # Create 2D overlay using the homography for this frame
        current_H = homography_list[fn]
        overlay_frame = create_2d_pitch_overlay(frame, map_img, current_H, alpha=0.4)
        
        # Optionally, add player tracking dots on the overlay
        # (You can comment this out if you don't want dots on the overlay)
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
        
        overlay_frames.append(overlay_frame)

    # 11) Save all three videos
    save_video(annotated_frames, output_path, fps=20.0)
    pitch_out = f"output_videos/{base}_pitch.avi"
    save_video(pitch_frames, pitch_out, fps=20.0)
    overlay_out = f"output_videos/{base}_overlay.avi"  # New overlay video
    save_video(overlay_frames, overlay_out, fps=20.0)
    
    print(f"Saved annotated video   → {output_path}")
    print(f"Saved pitch-only video  → {pitch_out}")
    print(f"Saved 2D overlay video  → {overlay_out}")


if __name__ == "__main__":
    main()