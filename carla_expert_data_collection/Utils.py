# 获取这个世界除了被控车辆的其他车辆
import carla
range_in_meters = 50
def gain_vehicle(world,vehicle):
    SurroundingVehicle_data=[]
    vehicle_location = vehicle.get_location()
    data=[]
    nearby_vehicles = world.get_actors().filter('vehicle.*')
    for nearby_vehicle in nearby_vehicles:
        if nearby_vehicle.id!= vehicle.id:  # 排除被控车辆本身
            nearby_vehicle_location = nearby_vehicle.get_location()
            # 计算与被控车辆的距离
            distance = vehicle_location.distance(nearby_vehicle_location)
            # 计算被控车辆与周围车辆的距离
            lateral_distance = abs(vehicle_location.x-nearby_vehicle_location.x)
            # 计算纵向距离
            longitudinal_distance = abs(vehicle_location.y-nearby_vehicle_location.y)
            data=[nearby_vehicle.id,distance,lateral_distance,longitudinal_distance]
            SurroundingVehicle_data.append(data)

    return SurroundingVehicle_data