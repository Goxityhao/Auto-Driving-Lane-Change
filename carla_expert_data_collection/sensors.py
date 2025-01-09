import carla


class CarlaVehicleInfo:
    def __init__(self,ControlVehicle):
        self.ControlVehicle = ControlVehicle

    def get_vehicle_data(self):
        """Retrieve data for a given vehicle."""
        location = self.ControlVehicle.get_location()
        velocity = self.ControlVehicle.get_velocity()

        # Convert velocity from m/s to km/h
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5 * 3.6

        # Retrieve waypoint to get lane information
        map = self.ControlVehicle.get_world().get_map()
        waypoint = map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_id = waypoint.lane_id if waypoint else None

        return {
            "location": location,
            "speed": speed,
            "lane_id": lane_id
        }
