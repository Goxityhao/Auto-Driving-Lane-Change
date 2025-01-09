import torch
import json


def load_config(config_path='./configs/general_config.json'):
    with open(config_path, 'r') as file:
        return json.load(file)


class Checkpoint:
    def __init__(self):
        self.config = load_config()
        self.modelsave=self.config["project"]["model_save"]

    def save_checkpoint(model, optimizer, epoch):
        """
        保存训练过程中模型的状态和优化器的状态。
        :param model: 训练中的模型，通常包括策略网络和Q网络
        :param optimizer: 模型的优化器
        :param epoch: 当前的训练轮次（epoch）
        :param filepath: 保存模型的路径
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),  # 保存模型的状态字典
            'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器的状态字典
        }
        torch.save(checkpoint, self.config["project"]["model_save"])
        print(f"Checkpoint saved at epoch {epoch} to {self.config["project"]["model_save"]}")

    def load_checkpoint(model, optimizer, filepath):
        """
        加载模型的状态和优化器的状态。
        :param model: 需要加载权重的模型
        :param optimizer: 需要加载状态的优化器
        :param filepath: 加载模型的路径
        :return: epoch，表示加载的训练轮次
        """
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型权重
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器状态
        epoch = checkpoint['epoch']

        print(f"Checkpoint loaded from {filepath}, starting from epoch {epoch}")

        return epoch