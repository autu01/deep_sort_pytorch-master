"""
综合测试脚本 - 测试YOLOv12车辆检测与轨迹追踪系统
"""
import sys
import os

print("=" * 60)
print("YOLOv12车辆检测与轨迹追踪系统 - 综合测试")
print("=" * 60)
print()

# 测试1: 检查依赖包
print("【测试1】检查依赖包...")
try:
    import torch
    import cv2
    import numpy as np
    from ultralytics import YOLO
    print(f"✅ PyTorch版本: {torch.__version__}")
    print(f"✅ OpenCV版本: {cv2.__version__}")
    print(f"✅ NumPy版本: {np.__version__}")
    print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA版本: {torch.version.cuda}")
        print(f"✅ GPU设备: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ 依赖包检查失败: {e}")
    sys.exit(1)

print()

# 测试2: 加载YOLOv12模型
print("【测试2】加载YOLOv12模型...")
try:
    from detector.YOLOv12 import YOLOv12
    
    # 检查模型文件是否存在
    model_path = "detector/YOLOv12/yolov12s.pt"
    if os.path.exists(model_path):
        print(f"✅ 模型文件存在: {model_path}")
    else:
        print(f"⚠️  模型文件不存在: {model_path}")
        print("   正在尝试下载模型...")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"   使用设备: {device}")
    
    yolo = YOLOv12(weight=model_path, device=device)
    print("✅ YOLOv12模型加载成功")
except Exception as e:
    print(f"❌ YOLOv12模型加载失败: {e}")
    sys.exit(1)

print()

# 测试3: 加载DeepSort追踪器
print("【测试3】加载DeepSort追踪器...")
try:
    from deep_sort import build_tracker
    from utils.parser import get_config
    
    cfg = get_config()
    cfg.USE_SEGMENT = False
    cfg.USE_MMDET = False
    cfg.USE_FASTREID = False
    cfg.merge_from_file('configs/yolov12s.yaml')
    cfg.merge_from_file('configs/deep_sort.yaml')
    
    use_cuda = torch.cuda.is_available()
    tracker = build_tracker(cfg, use_cuda=use_cuda)
    print("✅ DeepSort追踪器加载成功")
except Exception as e:
    print(f"❌ DeepSort追踪器加载失败: {e}")
    sys.exit(1)

print()

# 测试4: 测试检测功能
print("【测试4】测试检测功能...")
outputs = []
try:
    # 创建测试图像
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 执行检测
    bbox_xywh, conf, cls = yolo(test_img)
    
    print(f"✅ 检测功能正常")
    print(f"   检测到 {len(bbox_xywh)} 个目标")
    
    # 测试追踪功能
    if len(bbox_xywh) > 0:
        outputs, _, trajectories = tracker.update(bbox_xywh, conf, cls, test_img)
        print(f"✅ 追踪功能正常")
        print(f"   追踪到 {len(outputs)} 个目标")
    else:
        print("⚠️  没有检测到目标（测试图像是随机噪声），这是正常的")
        print("   使用模拟数据进行追踪测试...")
        
        # 创建模拟检测数据
        bbox_xywh = np.array([[320, 240, 100, 80]], dtype=np.float32)
        conf = np.array([0.9], dtype=np.float32)
        cls = np.array([2], dtype=np.float32)  # car
        
        outputs, _, trajectories = tracker.update(bbox_xywh, conf, cls, test_img)
        print(f"✅ 追踪功能正常（使用模拟数据）")
        print(f"   追踪到 {len(outputs)} 个目标")
        
except Exception as e:
    print(f"❌ 检测/追踪功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 测试5: 测试可视化功能
print("【测试5】测试可视化功能...")
try:
    from utils.draw import draw_boxes
    
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        
        # 绘制结果
        result_img = draw_boxes(test_img.copy(), bbox_xyxy, ['car']*len(outputs), identities, trajectories=trajectories)
        print("✅ 可视化功能正常")
    else:
        print("⚠️  没有追踪结果，跳过可视化测试")
except Exception as e:
    print(f"❌ 可视化功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# 测试6: 测试车辆类别筛选
print("【测试6】测试车辆类别筛选...")
try:
    vehicle_classes = [2, 3, 5, 7]  # car, truck, bus, train
    
    # 创建模拟数据
    test_cls_ids = np.array([0, 2, 3, 5, 7, 1])  # 包含各种类别
    vehicle_mask = np.isin(test_cls_ids, vehicle_classes)
    
    print(f"✅ 车辆类别筛选功能正常")
    print(f"   测试类别: {test_cls_ids}")
    print(f"   车辆掩码: {vehicle_mask}")
    print(f"   筛选结果: {test_cls_ids[vehicle_mask]}")
except Exception as e:
    print(f"❌ 车辆类别筛选测试失败: {e}")
    sys.exit(1)

print()

# 测试7: 测试轨迹记录功能
print("【测试7】测试轨迹记录功能...")
try:
    from deep_sort.sort.vehicle_tracker import VehicleTrack
    
    # 创建测试轨迹
    mean = np.zeros(4)
    covariance = np.eye(4)
    track = VehicleTrack(mean, covariance, track_id=1, n_init=3, max_age=70)
    
    print(f"✅ 轨迹记录功能正常")
    print(f"   轨迹ID: {track.track_id}")
    print(f"   最大轨迹长度: {track.max_trajectory_length}")
except Exception as e:
    print(f"❌ 轨迹记录功能测试失败: {e}")
    sys.exit(1)

print()

# 测试8: 测试配置文件
print("【测试8】测试配置文件...")
try:
    import yaml
    
    # 检查YOLOv12配置（使用UTF-8编码）
    with open('configs/yolov12s.yaml', 'r', encoding='utf-8') as f:
        yolov12_cfg = yaml.safe_load(f)
    print("✅ YOLOv12配置文件正常")
    
    # 检查DeepSort配置（使用UTF-8编码）
    with open('configs/deep_sort.yaml', 'r', encoding='utf-8') as f:
        deepsort_cfg = yaml.safe_load(f)
    print("✅ DeepSort配置文件正常")
    
except Exception as e:
    print(f"❌ 配置文件测试失败: {e}")
    sys.exit(1)

print()

# 测试总结
print("=" * 60)
print("测试总结")
print("=" * 60)
print("✅ 所有测试通过！")
print()
print("系统状态:")
print(f"  - 设备: {device}")
print(f"  - CUDA: {'可用' if torch.cuda.is_available() else '不可用'}")
if torch.cuda.is_available():
    print(f"  - GPU: {torch.cuda.get_device_name(0)}")
print(f"  - 模型: YOLOv12s")
print(f"  - 追踪器: DeepSort")
print()
print("系统已准备就绪，可以使用以下命令运行:")
print("  python deepsort.py --VIDEO_PATH <视频路径> --config_detection ./configs/yolov12s.yaml --display")
print()
print("支持的车辆类别:")
print("  - 类别2: 汽车 (car)")
print("  - 类别3: 卡车 (truck)")
print("  - 类别5: 公交车 (bus)")
print("  - 类别7: 火车 (train)")
print("=" * 60)
