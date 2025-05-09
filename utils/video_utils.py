import os
import cv2
import tempfile


def change_fps(input_path, output_path, fps=20.0, codec='mp4v'):
    """
    Re-encode a video file to the specified frames per second using OpenCV.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_path}")
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()


def read_video(video_path, target_fps=None):
    """
    Read a video file into a list of frames. Optionally re-encode to target_fps before reading.
    """
    # Validate path
    if not video_path:
        raise ValueError("Error: video_path is empty!")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Error: The file '{video_path}' does not exist.")

    # If a target FPS is given, re-encode to a temp file first
    if target_fps is not None:
        base, ext = os.path.splitext(video_path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_path = tmp.name
        tmp.close()
        change_fps(video_path, tmp_path, fps=target_fps)
        video_path = tmp_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path, fps=24.0, codec='XVID'):
    """
    Save a list of frames to a video file at the given fps.
    """
    if not output_video_frames:
        raise ValueError("Error: No frames to save.")
    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()