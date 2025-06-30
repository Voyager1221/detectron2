import cv2
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
import torch

# 1. Setup configuration for pre-trained Mask R-CNN
cfg = get_cfg()
cfg.MODEL.DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
predictor = DefaultPredictor(cfg)

# 2. Run inference on each image in images/
input_dir = "images"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_file)
    img = cv2.imread(img_path)

    outputs = predictor(img)
    
    # 3. Visualize predictions
    v = Visualizer(
        img[:, :, ::-1],
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        scale=1.0
    )

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_img = out.get_image()[:,:, ::-1] # back to BGR for cv2

    # 4. Save result
    out_path = os.path.join(output_dir, img_file)
    cv2.imwrite(out_path, result_img)

print("Inference complete! Check your outputs/ folder.")

