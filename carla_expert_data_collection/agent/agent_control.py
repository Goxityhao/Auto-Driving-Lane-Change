import carla
import json
import pygame
import time
import numpy as np
from datetime import datetime


def load_config(path):
    config_path = path
    with open(config_path, 'r') as file:
        return json.load(file)





# 导入自定义的控制策略和传感器
# from control_config import load_control_config
# from agent_config import load_agent_config
# from sensors import Camera, Lidar
# from controller import KeyboardController, BehaviorCloningController


class AgentControl:
    def __init__(self, client, world, vehicle, config_path="configs/agent_config.json"):
        # 读取配置文件
        self.config = load_config(path="./carla_expert_data_collection/configs/agent_config.json")
        self.control_config = load_config(path="./carla_expert_data_collection/configs/control_config.json")
        # 初始化变量
        self.client = client
        self.world = world
        self.vehicle = vehicle

        # 初始化传感器
        self.camera = None
        self.lidar = None
        self.controller = None

        # 配置传感器
        if self.config['sensor_config']['camera']['enabled']:
            self.camera = Camera(self.vehicle, self.config['sensor_config']['camera'])
        if self.config['sensor_config']['lidar']['enabled']:
            self.lidar = Lidar(self.vehicle, self.config['sensor_config']['lidar'])

        # 根据控制策略选择控制器
        self.setup_controller()

    def setup_controller(self):
        """
        根据控制策略选择合适的控制器
        """
        control_strategy = self.config['agent_behavior']['control_strategy']

        if control_strategy == "keyboard":
            self.controller = KeyboardController(self.vehicle, self.control_config)
        elif control_strategy == "behavior_cloning":
            model_path = self.config['control_method']['model_path']
            self.controller = BehaviorCloningController(self.vehicle, model_path)
        else:
            raise ValueError(f"Unsupported control strategy: {control_strategy}")

    def control_loop(self):
        """
        控制循环，接收键盘输入或行为克隆模型的控制指令
        """
        running = True
        clock = pygame.time.Clock()

        while running:
            clock.tick(30)  # 每秒30帧

            # 获取键盘输入或者行为克隆控制
            control = self.controller.get_control()

            # 应用控制
            self.vehicle.apply_control(control)

            # 如果启用日志，记录控制行为
            if self.config['behavior_logging']['enabled']:
                self.log_behavior(control)

            # 检查退出条件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # 清理
        self.cleanup()

    def log_behavior(self, control):
        """
        日志记录控制行为
        """
        if not self.config['behavior_logging']['enabled']:
            return

        log_data = {
            "timestamp": str(datetime.now()),
            "control": {
                "throttle": control.throttle,
                "steer": control.steer,
                "brake": control.brake,
                "hand_brake": control.hand_brake,
                "reverse": control.reverse
            }
        }

        with open(self.config['behavior_logging']['log_file'], "a") as log_file:
            log_file.write(json.dumps(log_data) + "\n")

    def cleanup(self):
        """
        清理操作，关闭传感器和其他资源
        """
        if self.camera:
            self.camera.cleanup()
        if self.lidar:
            self.lidar.cleanup()


if __name__ == "__main__":
    # 连接到Carla服务器
    client = carla.Client("localhost", 2000)
    client.set_timeout(10)

    # 加载世界和车辆
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("model3")[0]  # 选择特定的车辆模型
    spawn_point = carla.Transform(carla.Location(x=0, y=0, z=1), carla.Rotation(yaw=0))
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # 初始化智能体控制
    agent = AgentControl(client, world, vehicle)

    # 启动控制循环
    agent.control_loop()
