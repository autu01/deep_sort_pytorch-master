import numpy as np
import cv2
from .track import Track, TrackState


class VehicleTrack(Track):
    """
    扩展Track类，添加车辆轨迹记录功能
    """
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None, cls=None, mask=None):
        super().__init__(mean, covariance, track_id, n_init, max_age, feature, cls, mask)
        self.trajectory = []  # 轨迹点列表，每个点为 (x, y, timestamp)
        self.max_trajectory_length = 30  # 最大轨迹长度
        self.last_position = None  # 上一位置
        self.speed = 0  # 速度估计
        self.direction = None  # 行驶方向
        self.timestamp = 0  # 时间戳

    def update(self, kf, detection, timestamp=None):
        """
        更新轨迹信息
        """
        super().update(kf, detection)
        
        # 获取当前位置（中心点）
        current_position = self.mean[:2].copy()
        self.last_position = current_position
        
        # 添加轨迹点
        if timestamp is None:
            timestamp = self.timestamp
        self.trajectory.append((current_position[0], current_position[1], timestamp))
        
        # 限制轨迹长度
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
        
        # 更新时间戳
        self.timestamp = timestamp
        
        # 计算速度和方向
        self._calculate_speed_and_direction()

    def _calculate_speed_and_direction(self):
        """
        计算速度和方向
        """
        if len(self.trajectory) >= 2:
            # 取最近两个点计算
            (x1, y1, t1), (x2, y2, t2) = self.trajectory[-2], self.trajectory[-1]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            time_diff = t2 - t1
            
            if time_diff > 0:
                self.speed = distance / time_diff
                
                # 计算方向（角度）
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                self.direction = angle

    def get_trajectory(self):
        """
        获取轨迹点列表
        """
        return [(x, y) for x, y, t in self.trajectory]

    def draw_trajectory(self, image, color=(0, 255, 0), thickness=2):
        """
        在图像上绘制轨迹
        """
        trajectory = self.get_trajectory()
        if len(trajectory) >= 2:
            for i in range(1, len(trajectory)):
                cv2.line(image, 
                         (int(trajectory[i-1][0]), int(trajectory[i-1][1])),
                         (int(trajectory[i][0]), int(trajectory[i][1])),
                         color, thickness)
        return image


class VehicleTracker:
    """
    车辆轨迹追踪器，管理所有车辆的轨迹
    """
    def __init__(self, max_age=70, n_init=3):
        self.tracks = {}  # 存储所有车辆轨迹，key为track_id
        self.max_age = max_age
        self.n_init = n_init
        self.next_id = 1  # 下一个轨迹ID

    def update(self, detections, timestamp=None):
        """
        更新轨迹
        """
        # 这里需要与DeepSort的追踪器集成
        # 暂时留作接口
        pass

    def add_track(self, track):
        """
        添加新轨迹
        """
        self.tracks[track.track_id] = track

    def remove_track(self, track_id):
        """
        移除轨迹
        """
        if track_id in self.tracks:
            del self.tracks[track_id]

    def get_track(self, track_id):
        """
        获取轨迹
        """
        return self.tracks.get(track_id)

    def get_all_tracks(self):
        """
        获取所有活跃轨迹
        """
        return [track for track in self.tracks.values() if track.is_confirmed()]

    def draw_all_trajectories(self, image, color=(0, 255, 0), thickness=2):
        """
        绘制所有轨迹
        """
        for track in self.get_all_tracks():
            track.draw_trajectory(image, color, thickness)
        return image

    def get_traffic_statistics(self):
        """
        获取交通统计信息
        """
        active_tracks = self.get_all_tracks()
        total_vehicles = len(active_tracks)
        
        # 计算平均速度
        speeds = [track.speed for track in active_tracks if track.speed > 0]
        average_speed = np.mean(speeds) if speeds else 0
        
        # 统计方向分布
        directions = [track.direction for track in active_tracks if track.direction is not None]
        
        return {
            "total_vehicles": total_vehicles,
            "average_speed": average_speed,
            "directions": directions
        }
