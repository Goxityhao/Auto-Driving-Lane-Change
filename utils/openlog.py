import carla

def main():
    # 连接到 CARLA 服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 回放 `.log` 文件
    log_file = "my_simulation_01.log"
    print(f"Replaying file: {log_file}")

    # 回放文件
    client.replay_file(
        filename=log_file,
        start_time=0.0,      # 从头开始回放
        duration=0.0,        # 回放到结束
        follow_id=0,         # 不跟随任何对象
        replay_sensors=False # 不回放传感器数据
    )
    print("Replay started successfully.")

if __name__ == "__main__":
    main()