from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner

def main():
    # read video
    video_frames = read_video("input_videos/short-input-video.mp4", target_fps=20.0)

    #initialise tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path = "stubs/track_stubs.pkl")

    # print(f"Total frames tracked: {len(tracks['players'])}")

    # save cropped image of one player
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     cv2.imwrite(f"output_videos/cropped_image.jpg", cropped_image)

    #     break

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

    # team_assigner.assign_team_color(video_frames[0], 
    #                                 tracks['players'][0])
    
    # for each frame, assign each player to a team
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)

            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # save video
    save_video(output_video_frames, "output_videos/short-output-video.avi", fps=20.0)

if __name__ == "__main__":
    main()