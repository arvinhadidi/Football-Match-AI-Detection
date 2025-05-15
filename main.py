import os
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from tkinter import Tk, filedialog
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

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

    # Assign Player Teams
    team_assigner = TeamAssigner()

    # build a clean dict of only “real” players (exclude any referee IDs)
    first_frame_players = {
        pid: det
        for pid, det in tracks['players'][0].items()
        if pid not in tracks['referees'][0]
    }

    team_assigner.assign_team_color(video_frames[0],
                                    first_frame_players)
    
    # for each frame, assign each player to a team
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)

            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign ball to closest player
    player_assigner = PlayerBallAssigner()
    team_ball_control= []

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # if a player was assigned...
        if assigned_player != -1:
            # new parameter in our 'tracks' dictionary: a boolean of has_ball
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, output_path, fps=20.0)
    print(f"Saved annotated video to: {output_path}")

if __name__ == "__main__":
    main()