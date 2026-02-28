import cv2
import numpy as np
import os
import json
from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

print('Starting test...')

# 加载配置
print('Loading config...')
cfg = get_config()
cfg.USE_SEGMENT = False
cfg.USE_MMDET = False
cfg.USE_FASTREID = False
cfg.merge_from_file('configs/yolov12s.yaml')
cfg.merge_from_file('configs/deep_sort.yaml')
print('Config loaded successfully')

# 构建检测器和追踪器
print('Building detector and tracker...')
use_cuda = True
try:
    detector = build_detector(cfg, use_cuda=use_cuda, segment=False)
    print('Detector built successfully')
except Exception as e:
    print(f'Error building detector: {e}')
    exit(1)

try:
    tracker = build_tracker(cfg, use_cuda=use_cuda)
    print('Tracker built successfully')
except Exception as e:
    print(f'Error building tracker: {e}')
    exit(1)

# 测试图像路径
image_paths = ['demo/1.jpg', 'demo/2.jpg']
print(f'Testing images: {image_paths}')

# 车辆类别ID
vehicle_classes = [2, 3, 5, 7]  # car, truck, bus, train
print(f'Vehicle classes: {vehicle_classes}')

for image_path in image_paths:
    print(f'\nProcessing {image_path}...')
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f'Image {image_path} does not exist')
        continue
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f'Failed to read image {image_path}')
        continue
    print(f'Image shape: {img.shape}')
    
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 检测
    print('Running detection...')
    try:
        bbox_xywh, cls_conf, cls_ids = detector(im)
        print(f'Detection results: {len(bbox_xywh)} objects detected')
    except Exception as e:
        print(f'Error during detection: {e}')
        continue
    
    if len(bbox_xywh) > 0:
        print(f'Objects detected: {len(bbox_xywh)}')
        print(f'Class IDs: {cls_ids}')
        
        # 筛选车辆
        vehicle_mask = np.isin(cls_ids, vehicle_classes)
        print(f'Vehicle mask: {vehicle_mask}')
        
        bbox_xywh = bbox_xywh[vehicle_mask]
        cls_conf = cls_conf[vehicle_mask]
        cls_ids = cls_ids[vehicle_mask]
        
        print(f'Vehicles detected: {len(bbox_xywh)}')
        
        if len(bbox_xywh) > 0:
            # 追踪
            print('Running tracking...')
            try:
                outputs, _, trajectories = tracker.update(bbox_xywh, cls_conf, cls_ids, im)
                print(f'Tracking results: {len(outputs)} objects tracked')
            except Exception as e:
                print(f'Error during tracking: {e}')
                continue
            
            # 绘制结果
            if len(outputs) > 0:
                print(f'Objects tracked: {len(outputs)}')
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                cls = outputs[:, -2]
                
                # 类别名称
                with open('coco_classes.json', 'r') as f:
                    idx_to_class = json.load(f)
                names = [idx_to_class[str(int(label))] for label in cls]
                print(f'Tracked objects: {names}')
                
                # 绘制
                print('Drawing boxes and trajectories...')
                img = draw_boxes(img, bbox_xyxy, names, identities, trajectories=trajectories)
                
                # 保存结果
                output_path = f'output/{os.path.basename(image_path)}'
                os.makedirs('output', exist_ok=True)
                cv2.imwrite(output_path, img)
                print(f'Saved result to {output_path}')
            else:
                print('No vehicles tracked')
        else:
            print('No vehicles detected')
    else:
        print('No objects detected')

print('\nTest completed!')
