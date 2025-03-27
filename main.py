from utils import read_video, save_video
from trackers import Tracker

def main():
    # read video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    #initialise tracker
    tracker = Tracker("models/best.pt")

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path = "stubs/track_stubs.pkl")

    # save video
    save_video(video_frames, "output_videos/output_vid.avi")

if __name__ == "__main__":
    main()