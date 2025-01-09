import carla
import pygame
import time
import numpy as np
from load_config import load_config

json_file = "./agent_data.json"
config = load_config(json_file)

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA format
    surface = pygame.surfarray.make_surface(array[:, :, :3].swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()

def main():
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.load_world("Town02")

        # Enable synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Get the blueprint library
        blueprint_library = world.get_blueprint_library()

        # Get a vehicle blueprint
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]

        # Spawn the vehicle
        Location = carla.Location(config["ego_position"]["x"], config["ego_position"]["y"], config["ego_position"]["z"])
        Rotation = carla.Rotation(config["ego_rotation"]["pitch"], config["ego_rotation"]["yaw"], config["ego_rotation"]["roll"])
        spawn_points = carla.Transform(Location, Rotation)

        vehicle = world.spawn_actor(vehicle_bp, spawn_points)

        # Create a camera blueprint and attach it to the vehicle
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Adjust position relative to the vehicle
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Initialize pygame
        pygame.init()
        global display
        display = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("CARLA Vehicle Control")
        clock = pygame.time.Clock()

        # Control variables
        control = carla.VehicleControl()

        # Time tracking for position logging
        last_logged_time = time.time()

        # Attach image processing callback
        camera.listen(process_image)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Control logic
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                control.throttle = 1.0
            else:
                control.throttle = 0.0

            if keys[pygame.K_s]:
                control.brake = 1.0
            else:
                control.brake = 0.0

            if keys[pygame.K_a]:
                control.steer = max(control.steer - 0.05, -1.0)
            elif keys[pygame.K_d]:
                control.steer = min(control.steer + 0.05, 1.0)
            else:
                control.steer = 0.0

            vehicle.apply_control(control)

            # Output position and rotation every 2 seconds
            current_time = time.time()
            if current_time - last_logged_time >= 2.0:
                transform = vehicle.get_transform()
                location = transform.location
                rotation = transform.rotation
                print(f"Position: x={location.x}, y={location.y}, z={location.z}, Rotation: yaw={rotation.yaw}")
                last_logged_time = current_time

            # Tick the clock
            clock.tick(30)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Cleanup
        if camera:
            camera.stop()
            camera.destroy()
        if vehicle:
            vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        pygame.quit()

if __name__ == "__main__":
    main()
