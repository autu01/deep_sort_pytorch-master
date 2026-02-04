import cv2
import os
from tqdm import tqdm

# ########## 仅需修改这两个路径 ##########
img_dir = r"E:\vscode\deepsort-video\test\MOT16-06\img1"  # MOT16图片序列的文件夹路径
save_video_path = r"E:\vscode\deepsort-video\test\MOT16-06\mp4\MOT16-04.mp4"  # 合成后视频的保存路径+文件名
# #######################################

# 获取所有图片，按数字排序（关键，避免帧序混乱）
img_list = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
img_list.sort(key=lambda x: int(os.path.splitext(x)[0]))  # 按帧号数字排序

# 读取第一张图片，获取分辨率
first_img = cv2.imread(os.path.join(img_dir, img_list[0]))
h, w, _ = first_img.shape
# 视频编码器（Windows适配mp4），帧率30（和MOT16原数据集一致）
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(save_video_path, fourcc, 30, (w, h))

# 逐帧写入合成视频
for img_name in tqdm(img_list, desc="合成视频中"):
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        video_writer.write(img)

# 释放资源
video_writer.release()
cv2.destroyAllWindows()
print(f"视频合成完成！保存路径：{save_video_path}")