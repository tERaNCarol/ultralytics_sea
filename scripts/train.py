import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from ultralytics import YOLO

weight_addr = "/root/autodl-tmp/SeaDroneSee/weights/"
dataset_config_addr = "./ultralytics/cfg/datasets/"
yaml_addr = "./ultralytics/cfg/models/v8/"

model = YOLO(weight_addr + "yolov8s.pt")
# model = YOLO(yaml_addr + "yolov8s.yaml") 
model.train(data = dataset_config_addr + "SeaDroneSee.yaml", 
            epochs = 300, imgsz = 960, batch = 32,
            device = 0, workers = 0, save_json = True)
