import gym
import numpy as np
import carla
import random
import time
from gym import spaces
from collections import deque


class CarlaEnv(gym.Env):
    """
    自定义Carla环境，符合OpenAI Gym风格
    """

    def __init__(self, carla_client, host='localhost', port=2000, town='Town01', frame_skip=1):
        super(CarlaEnv, self).__init__()

        # Carla连接和仿真设置
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)

        # 设置仿真参数
        self.frame_skip = frame_skip
        self.actor = None  # 车辆对象
        self.sensor = None  # 传感器（例如相机、雷达等）

        # 环境状态空间和动作空间定义
        self.action_space = spaces.Discrete(4)  # 4个动作：加速、刹车、左转、右转
        self.observation_space = spaces.Box(low=np.array([-100, -100, -100, -100]),
                                            high=np.array([100, 100, 100, 100]), dtype=np.float32)  # 自定义状态空间

        # 环境变量
        self.current_step = 0
        self.done = False
        self.vehicle_position = np.array([0.0, 0.0, 0.0, 0.0])  # 初始位置

    def reset(self):
        """
        重置环境，重新初始化仿真状态。
        """
        self.done = False
        self.current_step = 0

        # 清理现有的仿真对象
        if self.actor:
            self.actor.destroy()
        if self.sensor:
            self.sensor.destroy()

        # 重新生成车辆和传感器
        self.actor = self.spawn_vehicle()
        self.sensor = self.add_sensor()

        # 获取初始状态
        initial_state = self.get_state()
        return initial_state

    def step(self, action):
        """
        执行一个动作，更新环境状态并返回新的状态、奖励、完成标志和额外信息。
        :param action: 动作（0:加速, 1:刹车, 2:左转, 3:右转）
        :return: new_state, reward, done, info
        """
        # 执行动作
        self.apply_action(action)

        # 更新仿真
        self.world.tick()

        # 获取新的状态
        new_state = self.get_state()

        # 计算奖励
        reward = self.compute_reward()

        # 判断是否结束
        if self.current_step >= 1000:  # 最大步数限制
            self.done = True

        # 增加步数
        self.current_step += 1

        return new_state, reward, self.done, {}

    def render(self, mode='human'):
        """
        渲染环境（可选）
        """
        if mode == 'human':
            # 这里只是一个简单的渲染示例，实际情况可以用摄像头等传感器进行渲染
            print(f"Vehicle position: {self.vehicle_position}")

    def close(self):
        """
        关闭环境并清理资源
        """
        if self.actor:
            self.actor.destroy()
        if self.sensor:
            self.sensor.destroy()
        print("Environment closed.")

    def spawn_vehicle(self):
        """
        在Carla中生成车辆
        :return: 车辆对象
        """
        blueprint = self.world.get_blueprint_library().find('vehicle.tesla.model3')
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        vehicle = self.world.spawn_actor(blueprint, spawn_point)
        return vehicle

    def add_sensor(self):
        """
        添加传感器（例如相机）来获取车辆状态信息
        :return: 传感器对象
        """
        # 相机传感器
        blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        spawn_point = carla.Transform(carla.Location(x=1.5, z=2.5))
        camera = self.world.spawn_actor(blueprint, spawn_point, attach_to=self.actor)
        camera.listen(lambda image: self.process_image(image))  # 这里处理图像
        return camera

    def process_image(self, image):
        """
        处理传感器图像，转换为状态空间信息
        :param image: 从传感器获取的图像
        """
        # 你可以在这里添加图像处理逻辑，例如将图像转换为灰度图，提取特征等
        self.vehicle_position = np.array([image.timestamp, image.frame, image.width, image.height])

    def apply_action(self, action):
        """
        根据动作选择应用加速、刹车、左转或右转
        :param action: 动作（0:加速, 1:刹车, 2:左转, 3:右转）
        """
        control = carla.VehicleControl()

        if action == 0:  # 加速
            control.throttle = 1.0
            control.steer = 0
        elif action == 1:  # 刹车
            control.brake = 1.0
            control.steer = 0
        elif action == 2:  # 左转
            control.steer = -1.0
        elif action == 3:  # 右转
            control.steer = 1.0

        self.actor.apply_control(control)

    def get_state(self):
        """
        获取当前状态
        :return: 当前状态（例如车辆的位置、速度等信息）
        """
        location = self.actor.get_location()
        velocity = self.actor.get_velocity()

        state = np.array([location.x, location.y, velocity.x, velocity.y])
        return state

    def compute_reward(self):
        """
        根据当前状态计算奖励值
        :return: 奖励值
        """
        # 你可以根据车辆是否靠近目标，或者是否发生碰撞来计算奖励
        reward = -np.linalg.norm(self.vehicle_position[:2])  # 假设距离越远奖励越低
        return reward

# 如果你在主程序中直接使用环境时，可以这样创建环境：
# carla_client = carla.Client('localhost', 2000)
# env = CarlaEnv(carla_client)
