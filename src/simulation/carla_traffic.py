import carla
import random
import time
import numpy as np


class CarlaTraffic:
    def __init__(self, world, spawn_points=None, traffic_density=0.5):
        """
        Carla 交通流控制类
        :param world: Carla 世界实例
        :param spawn_points: 车辆生成点列表，如果为空，则自动选择
        :param traffic_density: 交通密度，范围为 [0, 1]，1 为最大密度
        """
        self.world = world
        self.spawn_points = spawn_points or self.world.get_map().get_spawn_points()
        self.traffic_density = traffic_density  # 交通密度
        self.vehicles = []  # 车辆列表
        self.walkers = []  # 行人列表
        self.client = self.world.get_client()

    def spawn_vehicles(self, num_vehicles=None):
        """
        生成指定数量的交通车辆
        :param num_vehicles: 生成的车辆数量，如果为 None，则根据交通密度生成
        """
        num_vehicles = num_vehicles or int(len(self.spawn_points) * self.traffic_density)
        blueprint_library = self.world.get_blueprint_library()

        for _ in range(num_vehicles):
            blueprint = random.choice(blueprint_library.filter('vehicle'))
            spawn_point = random.choice(self.spawn_points)
            vehicle = self.world.spawn_actor(blueprint, spawn_point)

            # 设置车辆的控制（例如：自动驾驶、速度限制等）
            vehicle.set_autopilot(True)  # 启用自动驾驶
            self.vehicles.append(vehicle)
            print(f"Spawned vehicle at {spawn_point.location}")

    def spawn_walkers(self, num_walkers=10):
        """
        生成指定数量的行人
        :param num_walkers: 生成的行人数量
        """
        walker_blueprint = self.world.get_blueprint_library().filter('walker.pedestrian.*')

        for _ in range(num_walkers):
            spawn_point = self._get_random_spawn_point()
            walker = self.world.spawn_actor(random.choice(walker_blueprint), spawn_point)
            self.walkers.append(walker)
            print(f"Spawned walker at {spawn_point.location}")

    def _get_random_spawn_point(self):
        """
        随机选择一个生成点
        :return: Carla.Transform
        """
        spawn_point = random.choice(self.spawn_points)
        return carla.Transform(carla.Location(
            random.uniform(spawn_point.location.x - 10, spawn_point.location.x + 10),
            random.uniform(spawn_point.location.y - 10, spawn_point.location.y + 10),
            spawn_point.location.z))

    def set_traffic_lights(self):
        """
        设置交通灯信号
        """
        for light in self.world.get_actors().filter('traffic.light'):
            light.set_state(carla.TrafficLightState.Green)  # 设置所有交通灯为绿灯
            print("Traffic light set to green.")

    def control_traffic(self, manual_override=False):
        """
        控制交通流量，手动接管时可以通过 override 参数控制
        :param manual_override: 手动接管交通控制
        """
        if manual_override:
            print("Manual override activated, control traffic flow manually.")
            # 手动控制，例如车速、红绿灯等
            self.set_traffic_lights()  # 假设手动接管时需要设置交通灯
        else:
            print("Traffic control is automatic.")
            # 在自动模式下，可以控制交通流量，例如设定交通密度等
            # 可以根据交通密度动态调整交通流

    def run_traffic(self):
        """
        启动交通流（包括车辆和行人）
        """
        print("Starting traffic flow...")

        # 启动车辆
        self.spawn_vehicles()

        # 启动行人
        self.spawn_walkers()

        # 设置交通灯
        self.set_traffic_lights()

        # 持续更新，模拟交通流
        self.world.tick()

    def stop_traffic(self):
        """
        停止所有交通流（车辆、行人、交通灯）
        """
        print("Stopping traffic flow...")
        for vehicle in self.vehicles:
            vehicle.destroy()
        for walker in self.walkers:
            walker.destroy()
        self.vehicles.clear()
        self.walkers.clear()

    def clean_up(self):
        """
        清理所有交通流相关对象
        """
        print("Cleaning up traffic...")
        self.stop_traffic()

    def reset_traffic(self):
        """
        重置交通环境
        """
        print("Resetting traffic environment...")
        self.clean_up()
        self.run_traffic()


# 主程序示例
if __name__ == "__main__":
    try:
        # 连接到Carla模拟器
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        # 创建Carla交通控制实例
        traffic_manager = CarlaTraffic(world)

        # 启动交通流
        traffic_manager.run_traffic()

        # 运行一段时间后清理
        time.sleep(10)

        # 停止交通流
        traffic_manager.stop_traffic()

    except Exception as e:
        print(f"Error: {e}")
