import cv2


video_path = "input_videos/08fd33_4.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open: {video_path}")
else:
    print("Video opened successfully!")

cap.release()