import carla
import numpy as np
import cv2
from collections import deque
from gym import spaces


class Sensor:
    def __init__(self, vehicle, sensor_type, transform, attributes=None):
        """
        通用传感器基类，定义了与 Carla 中的传感器的基本交互方法
        :param vehicle: 关联的车辆
        :param sensor_type: 传感器类型 ('camera', 'lidar', 'depth', 'collision')
        :param transform: 传感器的位置与方向
        :param attributes: 传感器的额外属性（如摄像头的分辨率、Lidar的扫描角度等）
        """
        self.vehicle = vehicle
        self.sensor_type = sensor_type
        self.transform = transform
        self.attributes = attributes or {}
        self.sensor = None
        self.data = None
        self.listener = None
        self.image_queue = deque(maxlen=10)  # 用于存储图像数据的队列

    def _create_sensor(self):
        """
        根据传感器类型，创建Carla中的传感器
        """
        blueprint_library = self.vehicle.get_world().get_blueprint_library()

        if self.sensor_type == 'camera':
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.attributes.get('width', 800)))
            camera_bp.set_attribute('image_size_y', str(self.attributes.get('height', 600)))
            camera_bp.set_attribute('fov', str(self.attributes.get('fov', 90)))
            self.sensor = self.vehicle.get_world().spawn_actor(camera_bp, self.transform, attach_to=self.vehicle)
            self.sensor.listen(self._on_camera_data)
        elif self.sensor_type == 'lidar':
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', str(self.attributes.get('range', 100)))
            self.sensor = self.vehicle.get_world().spawn_actor(lidar_bp, self.transform, attach_to=self.vehicle)
            self.sensor.listen(self._on_lidar_data)
        elif self.sensor_type == 'depth':
            depth_bp = blueprint_library.find('sensor.camera.depth')
            depth_bp.set_attribute('image_size_x', str(self.attributes.get('width', 800)))
            depth_bp.set_attribute('image_size_y', str(self.attributes.get('height', 600)))
            depth_bp.set_attribute('fov', str(self.attributes.get('fov', 90)))
            self.sensor = self.vehicle.get_world().spawn_actor(depth_bp, self.transform, attach_to=self.vehicle)
            self.sensor.listen(self._on_depth_data)
        elif self.sensor_type == 'collision':
            collision_bp = blueprint_library.find('sensor.other.collision')
            self.sensor = self.vehicle.get_world().spawn_actor(collision_bp, self.transform, attach_to=self.vehicle)
            self.sensor.listen(self._on_collision_data)
        else:
            raise ValueError(f"Unsupported sensor type: {self.sensor_type}")

    def _on_camera_data(self, image):
        """
        摄像头数据回调函数
        :param image: Carla图像对象
        """
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_data = image_data.reshape((image.height, image.width, 4))
        image_data = image_data[:, :, :3]  # RGB 数据
        self.image_queue.append(image_data)
        self.data = image_data

    def _on_lidar_data(self, lidar_data):
        """
        激光雷达数据回调函数
        :param lidar_data: Carla激光雷达数据
        """
        lidar_points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        lidar_points = lidar_points.reshape((len(lidar_points) // 4, 4))
        self.data = lidar_points

    def _on_depth_data(self, depth_image):
        """
        深度相机数据回调函数
        :param depth_image: Carla深度图像数据
        """
        depth_data = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
        depth_data = depth_data.reshape((depth_image.height, depth_image.width, 4))
        depth_data = depth_data[:, :, :3]  # 只取 RGB 数据
        self.data = depth_data

    def _on_collision_data(self, collision_info):
        """
        碰撞传感器数据回调函数
        :param collision_info: 碰撞信息
        """
        self.data = collision_info

    def start(self):
        """
        启动传感器并开始接收数据
        """
        self._create_sensor()

    def stop(self):
        """
        停止传感器并销毁
        """
        if self.sensor:
            self.sensor.destroy()
            self.sensor = None

    def get_data(self):
        """
        获取传感器数据
        :return: 传感器当前数据
        """
        return self.data

    def get_image_queue(self):
        """
        获取图像数据队列
        :return: 存储图像的队列
        """
        return self.image_queue

    def get_last_image(self):
        """
        获取最后一帧图像数据
        :return: 最后一帧图像
        """
        if self.image_queue:
            return self.image_queue[-1]
        return None


class CameraSensor(Sensor):
    def __init__(self, vehicle, transform, width=800, height=600, fov=90):
        """
        初始化摄像头传感器
        :param vehicle: 关联的车辆
        :param transform: 传感器的位置与方向
        :param width: 图像宽度
        :param height: 图像高度
        :param fov: 视野范围（度）
        """
        attributes = {'width': width, 'height': height, 'fov': fov}
        super().__init__(vehicle, sensor_type='camera', transform=transform, attributes=attributes)


class LidarSensor(Sensor):
    def __init__(self, vehicle, transform, range=100):
        """
        初始化激光雷达传感器
        :param vehicle: 关联的车辆
        :param transform: 传感器的位置与方向
        :param range: 激光雷达的最大测距范围
        """
        attributes = {'range': range}
        super().__init__(vehicle, sensor_type='lidar', transform=transform, attributes=attributes)


class DepthCameraSensor(Sensor):
    def __init__(self, vehicle, transform, width=800, height=600, fov=90):
        """
        初始化深度相机传感器
        :param vehicle: 关联的车辆
        :param transform: 传感器的位置与方向
        :param width: 图像宽度
        :param height: 图像高度
        :param fov: 视野范围（度）
        """
        attributes = {'width': width, 'height': height, 'fov': fov}
        super().__init__(vehicle, sensor_type='depth', transform=transform, attributes=attributes)


class CollisionSensor(Sensor):
    def __init__(self, vehicle, transform):
        """
        初始化碰撞传感器
        :param vehicle: 关联的车辆
        :param transform: 传感器的位置与方向
        """
        super().__init__(vehicle, sensor_type='collision', transform=transform)
