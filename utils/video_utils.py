import os
import cv2

def read_video(video_path):

    # Check if video_path is empty
    if not video_path:
        raise ValueError("Error: video_path is empty!")

    # Check if the file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Error: The file '{video_path}' does not exist.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    frames = []
    while True:
        ret, frame = cap.read() # ret is a flag that says if the video has ended
        if not ret:
            break
        frames.append(frame) # add each frame to list of frames
    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # path of video as a string, the file output type, frames per second, width, height
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()