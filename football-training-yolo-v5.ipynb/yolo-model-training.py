#Get Dataset

from roboflow import Roboflow
rf = Roboflow(api_key="pjzoOBDfQrjsOlZkJXJO")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")