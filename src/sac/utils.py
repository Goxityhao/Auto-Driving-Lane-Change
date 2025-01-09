import torch
import os


# 保存模型的函数
def save_model(model, filename, directory="models"):
    """
    保存模型到指定路径。
    :param model: 要保存的模型
    :param filename: 保存的文件名
    :param directory: 保存模型的目录，默认为 "models"
    """
    # 如果目标目录不存在，则创建该目录
    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)

    # 保存模型
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


# 获取设备（CPU或GPU）
def get_device():
    """
    获取当前的设备（GPU或CPU）。
    :return: 返回 'cuda' 或 'cpu'，表示使用的设备类型
    """
    # 如果有GPU可用，则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
