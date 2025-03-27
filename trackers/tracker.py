from ultralytics import YOLO
import supervision as sv
import pickle
import os

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        # run thru all frames of video, in steps of eg. 20 
        for i in range (0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            break
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        # will run after first time
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks


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
            print(class_names)

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
                tracks["players"][frame_num][track_id] = {"bbox":bbox}

            if class_id == class_names_inverted["goalkeeper"]:
                tracks["goalkeepers"][frame_num][track_id] = {"bbox":bbox}

            if class_id == class_names_inverted["referee"]:
                tracks["referees"][frame_num][track_id] = {"bbox":bbox}

        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            class_id = frame_detection[3]

            if class_id == class_names_inverted["ball"]:
                tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks,f)

        return tracks

        
