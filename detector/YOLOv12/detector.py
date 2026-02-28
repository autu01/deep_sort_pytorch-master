import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv12 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics import YOLO
from ultralytics.utils import ops


class YOLOv12(object):
    def __init__(self, weight='yolov12s.pt', data='ultralytics/cfg/datasets/coco.yaml', imgsz=[640, 640],
                 conf_thres=0.25, nms_thres=0.45, max_det=1000, device='cuda:0'):
        super().__init__()
        self.device = device
        self.model = YOLO(weight)
        self.model.to(device)
        self.class_names = self.model.names
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.max_det = max_det

    def __call__(self, im0, augment=False, save_result=False):
        # im shape is [H, W, 3] and RGB
        # model inference
        results = self.model.predict(
            im0, 
            imgsz=self.imgsz, 
            conf=self.conf_thres, 
            iou=self.nms_thres, 
            max_det=self.max_det,
            augment=augment,
            save=save_result,
            device=self.device
        )
        
        # postprocess det
        pred = results[0].boxes
        if len(pred) == 0:
            return np.array([]), np.array([]), np.array([])
        
        xyxy = pred.xyxy.cpu().numpy()
        conf = pred.conf.cpu().numpy()
        cls = pred.cls.cpu().numpy()
        
        # convert xyxy to xywh
        xywh = []
        for box in xyxy:
            x1, y1, x2, y2 = box
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            xywh.append([x, y, w, h])
        xywh = np.array(xywh)
        
        if save_result is True:
            im0 = results[0].plot()
        
        return (xywh, conf, cls) if not save_result else (xywh, conf, cls, im0)


def demo():
    yolo = YOLOv12(weight='yolov12s.pt')
    root = "./ultralytics/assets"
    files = [os.path.join(root, file) for file in os.listdir(root) if file.endswith('.jpg')]
    for filename in files:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox, cls_conf, cls_ids, img_ = yolo(img, save_result=True)
        # imshow
        cv2.namedWindow("yolo")
        cv2.imshow("yolo", img_[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo()