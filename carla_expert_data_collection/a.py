import carla
import pygame
import numpy as np
import time

# 初始化 Pygame 控制键位的映射
def get_keyboard_control(keys):
    control = carla.VehicleControl()
    if keys[pygame.K_w]:
        control.throttle = 1.0
    if keys[pygame.K_s]:
        control.brake = 1.0
    if keys[pygame.K_a]:
        control.steer = -1.0
    if keys[pygame.K_d]:
        control.steer = 1.0
    return control

def main():
    # 初始化 Pygame
    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA Keyboard Control")
    clock = pygame.time.Clock()

    # 连接到 CARLA 服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 获取车辆 Blueprint
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]

    # 生成车辆
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # 设置固定视角摄像头
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=-5, z=2.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # 渲染相机画面
    image_surface = None
    def process_image(image):
        nonlocal image_surface
        # 转换 raw_data 为 NumPy 数组并转换为 RGB 格式
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA 格式
        array = array[:, :, :3][:, :, ::-1]  # 转为 RGB 格式
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    camera.listen(lambda image: process_image(image))

    try:
        running = True
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 获取键盘输入
            keys = pygame.key.get_pressed()
            control = get_keyboard_control(keys)
            vehicle.apply_control(control)

            # 渲染 Pygame 窗口
            if image_surface:
                display.blit(image_surface, (0, 0))
            pygame.display.flip()
            clock.tick(30)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")

    finally:
        # 销毁演员
        camera.destroy()
        vehicle.destroy()

        # 退出 Pygame
        pygame.quit()

if __name__ == "__main__":
    main()
