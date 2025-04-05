from utils import read_video, save_video
from trackers import Tracker

def main():
    # read video
    video_frames = read_video("input_videos/short-input-video.mp4")

    #initialise tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path = "stubs/track_stubs.pkl")
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    print(f"Total frames tracked: {len(tracks['players'])}")


    # save video
    save_video(output_video_frames, "output_videos/short_output_vid.avi")

if __name__ == "__main__":
    main()