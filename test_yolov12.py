import cv2
import numpy as np
from detector.YOLOv12 import YOLOv12

# 加载YOLOv12模型
yolo = YOLOv12(weight='detector/YOLOv12/yolov12s.pt', device='cuda:0')

# 测试图像路径
image_paths = ['demo/1.jpg', 'demo/2.jpg']

# 车辆类别ID
vehicle_classes = [2, 3, 5, 7]  # car, truck, bus, train

for image_path in image_paths:
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 检测
    bbox_xywh, conf, cls = yolo(img)
    
    # 筛选车辆
    vehicle_mask = np.isin(cls, vehicle_classes)
    vehicle_bboxes = bbox_xywh[vehicle_mask]
    vehicle_conf = conf[vehicle_mask]
    vehicle_cls = cls[vehicle_mask]
    
    # 转换为xyxy格式
    vehicle_bboxes_xyxy = []
    for bbox in vehicle_bboxes:
        x, y, w, h = bbox
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        vehicle_bboxes_xyxy.append([x1, y1, x2, y2])
    
    # 绘制 bounding boxes
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i, bbox in enumerate(vehicle_bboxes_xyxy):
        x1, y1, x2, y2 = bbox
        conf_score = vehicle_conf[i]
        cls_id = int(vehicle_cls[i])
        
        # 绘制矩形
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标签
        label = f'Vehicle {cls_id}: {conf_score:.2f}'
        cv2.putText(img_bgr, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow(f'Test {image_path}', img_bgr)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print('测试完成！')
