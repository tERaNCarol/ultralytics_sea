import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from ultralytics import YOLO

weight_addr = "./ultralytics/cfg/models/weights/"
dataset_config_addr = "./ultralytics/cfg/datasets/"

# model = YOLO(weight_addr + "sea_s1.pt")
model = YOLO("/root/ultralytics_sea/runs/detect/train13/weights/best.pt")

metrics = model.val(data = dataset_config_addr + "SeaDroneSee.yaml")
metrics.box.map
metrics.box.map50
metrics.box.maps
