import carla
import random


def generate_random_integer(lower, upper):
    """
    在输入的下限和上限中随机输出一个随机整数，这个随机数位于这两个数之间。

    Parameters:
        lower (int): 边界下限。
        upper (int): 边界上限。

    Returns:
        int: 返回上限和下限之间的一个随机整数。

    Raises:
        ValueError: 若下限数值大于上限，则报错。
    """
    if lower > upper:
        raise ValueError("The lower bound must be less than or equal to the upper bound.")
    return random.randint(lower, upper)


class CarGeneray:
    def __init__(self, NmuCar, InitialPosition, car_name, distance, world):
        self.spawn_x = InitialPosition[0]
        self.spawn_y = InitialPosition[1]
        self.spawn_z = InitialPosition[2]
        self.pitch = InitialPosition[3]
        self.yaw = InitialPosition[4]
        self.roll = InitialPosition[5]
        self.car_name = car_name
        self.num_vehicles = NmuCar
        self.distance = 6
        self.world = world
        self.initial_velocity = carla.Vector3D(15, 0, 0)

    def create_vehicle_in_line(self):
        """
                在直线上生成车辆。
                :param blueprint_name: 车辆蓝图名称
                :param distance_between: 每辆车之间的距离
                :return: 生成的车辆列表
                """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprint = blueprint_library.find(self.car_name)
        vehicle_list = []
        for i in range(self.num_vehicles):
            # distance=generate_random_integer(self.distance[0],self.distance[1])
            spawn_point = carla.Transform(
                carla.Location(
                    x=self.spawn_x + i * self.distance,
                    y=self.spawn_y,
                    z=self.spawn_z
                ),

                carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll)
            )
            vehicle = self.world.try_spawn_actor(vehicle_blueprint, spawn_point)
            vehicle_phys_control = vehicle.get_physics_control()
            vehicle_phys_control.velocity = self.initial_velocity
            vehicle.apply_physics_control(vehicle_phys_control)
            if vehicle:
                # print(f"Spawned vehicle {vehicle.id} at position {self.spawn_y - i * self.distance} meters.")
                vehicle_list.append(vehicle)
            else:
                print(f"Failed to spawn vehicle at {self.spawn_y - i * self.distance} meters.")
        return vehicle_list
