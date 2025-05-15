import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pickle
import pandas as pd
import os
import sys 
# sys.path.append('../')
# from utils import get_center_of_bbox, get_bbox_width
from utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        # 1) Build a list of exactly 4 numbers per frame,
        #    using NaN when the ball was not detected.
        bbox_list = []
        for frame_dict in ball_positions:
            bbox = frame_dict.get(1, {}).get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                bbox_list.append(bbox)
            else:
                bbox_list.append([np.nan, np.nan, np.nan, np.nan])

        # 2) Create a DataFrame with columns x1,y1,x2,y2
        df = pd.DataFrame(bbox_list, columns=["x1", "y1", "x2", "y2"])

        # 3) Interpolate missing values then back‑ and forward‑fill edges
        df = df.interpolate().bfill().ffill()

        # 4) Turn it back into your list‑of‑dicts format
        smoothed = []
        for row in df.to_numpy().tolist():
            smoothed.append({1: {"bbox": row}})
        return smoothed


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        # run thru all frames of video, in steps of eg. 20 
        for i in range (0, len(frames), batch_size):
            print(f"Processing frames {i} to {min(i+batch_size, len(frames))}")  # Debug print
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.3) # CHANGED CONF FROM 0.1 TO 0.3
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        detections = self.detect_frames(frames)

        # a dictionary of lists of dictionaries
        tracks = {
            "players":[], # each entry in this list will have output for one frame
            "goalkeepers":[],
            "referees":[],
            "ball":[]

        }

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inverted = {v:k for k,v in class_names.items()}
            # print(class_names)

            #convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # # Convert GoalKeeper to player object
            # for object_ind , class_id in enumerate(detection_supervision.class_id):
            #     if cls_names[class_id] == "goalkeeper":
            #         detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # add empty dictionary in each list, in the tracks dictionary
            tracks["players"].append({})
            tracks["goalkeepers"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_names_inverted["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if class_id == class_names_inverted["goalkeeper"]:
                    tracks["goalkeepers"][frame_num][track_id] = {"bbox": bbox}
                if class_id == class_names_inverted["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                if class_id == class_names_inverted["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_rectangle(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # cv2.ellipse(
        #     frame,
        #     center=(x_center,y2),
        #     axes=(int(width), int(0.35*width)),
        #     angle=0.0,
        #     startAngle=-45,
        #     endAngle=235,
        #     color = color,
        #     thickness=2,
        #     lineType=cv2.LINE_4
        # )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
    def draw_triangle(self,frame,bbox,color):
        y= bbox[1]
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ], dtype=np.int32)
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_annotations(self,video_frames, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # if frame_num >= len(tracks["players"]):  
            #     print(f"Skipping frame {frame_num} (out of range)")
            #     continue

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_rectangle(frame, player["bbox"],color, track_id)

                # draw red triangle on player who has ball
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_rectangle(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))

            output_video_frames.append(frame)

        return output_video_frames


        
