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
from homography.homography_accumulator import HomographyAccumulator

with open("calib/map_keypoints.json") as f:
    map_keypoints = json.load(f)

with open("calib/shortened_map_keypoints.json") as f:
    shortened_map_keypoints = json.load(f)

def load_map_keypoints(json_path="calib/map_keypoints.json"):
    """Returns a dict mapping label→[x,y] in map-image coordinates."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

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

    # Initialize the homography accumulator
    homography_accumulator = HomographyAccumulator(
        max_frames=15,           # Keep keypoints from last 15 frames
        min_keypoints=4,         # Need at least 4 points for homography
        stability_threshold=8.0, # Points within 8 pixels are considered same
        confidence_decay=0.92    # Older frames have less influence
    )

    # Homography + warp loop with accumulation
    DETECT_KEYPOINTS_EVERY = 20   # Detect keypoints every 5 frames (more frequent)
    RECALIBRATE_EVERY = 20       # Consider updating homography every 10 frames
    
    warped_player_info = []
    warped_ball_info = []
    fh, fw = video_frames[0].shape[:2]

    for fn, frame in enumerate(video_frames):
        # Detect keypoints more frequently to build up database
        if fn % DETECT_KEYPOINTS_EVERY == 0:
            try:
                image_pts, labels = match_keypoints_to_areas(frame)
                
                # Filter out edge points (same as before)
                filtered_pts, filtered_labels, confidences = [], [], []
                for (x, y), lbl in zip(image_pts, labels):
                    if 5 < x < fw-5 and 5 < y < fh-5 and lbl in shortened_map_keypoints:
                        filtered_pts.append((x, y))
                        filtered_labels.append(lbl)
                        confidences.append(1.0)  # You could add actual confidence scores here
                
                # Add keypoints to accumulator
                if filtered_pts:
                    homography_accumulator.add_frame_keypoints(
                        fn, filtered_pts, filtered_labels, confidences
                    )
                    print(f"Frame {fn}: Added {len(filtered_pts)} keypoints to accumulator")
                
            except Exception as e:
                print(f"Frame {fn}: Failed to detect keypoints: {e}")
        
        # Check if we should update homography
        if fn % RECALIBRATE_EVERY == 0 and fn > 0:
            if homography_accumulator.should_update_homography(fn):
                new_H = homography_accumulator.compute_accumulated_homography(
                    shortened_map_keypoints, fn
                )
                if new_H is not None:
                    # Optionally add debug visualization
                    debug_info = homography_accumulator.get_debug_info()
                    print(f"Frame {fn}: Updated homography using data from {debug_info['frames_in_memory']} frames")
                    print(f"Stable keypoints: {debug_info['keypoint_counts']}")
        
        # Get current best homography (whether updated this frame or not)
        H = homography_accumulator.get_current_homography()
        
        # Warp players and ball using current homography (same as before)
        frame_players = []
        for pid, det in tracks["players"][fn].items():
            cx, cy = get_center_of_bbox(det["bbox"])
            wx, wy = warp_point((cx, cy), H)
            frame_players.append((pid, (wx, wy)))

        bb = tracks["ball"][fn].get(1, {}).get("bbox")
        if bb:
            bx, by = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
            ball_pt = warp_point((bx, by), H)
        else:
            ball_pt = (np.nan, np.nan)

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


    # 11) Save both videos
    save_video(annotated_frames, output_path, fps=20.0)
    pitch_out = f"output_videos/{base}_pitch.avi"
    save_video(pitch_frames, pitch_out, fps=20.0)
    print(f"Saved annotated video   → {output_path}")
    print(f"Saved pitch-only video  → {pitch_out}")


if __name__ == "__main__":
    main()