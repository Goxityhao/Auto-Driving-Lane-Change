import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BehaviorCloningDataset(Dataset):
    """
    自定义的数据集类，用于加载行为克隆训练的数据。
    数据集包含状态和动作对。
    """

    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: 存放数据的目录，假设数据是以.npy文件存储的
        :param transform: 数据变换函数，可用于预处理
        """
        self.data_dir = data_dir
        self.transform = transform

        # 加载数据
        self.states = []
        self.actions = []

        # 假设数据以 numpy 格式存储
        state_file = os.path.join(data_dir, 'states.npy')
        action_file = os.path.join(data_dir, 'actions.npy')

        # 加载数据（假设状态和动作文件均为 .npy 格式）
        if os.path.exists(state_file) and os.path.exists(action_file):
            self.states = np.load(state_file)  # 状态数据，形状为 (N, state_dim)
            self.actions = np.load(action_file)  # 动作数据，形状为 (N, action_dim)
        else:
            raise FileNotFoundError("State or action file not found.")

        # 转换为 Tensor
        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.float32)

    def __len__(self):
        """返回数据集的大小"""
        return len(self.states)

    def __getitem__(self, idx):
        """返回单个样本的数据，包括状态和对应的动作"""
        state = self.states[idx]
        action = self.actions[idx]

        if self.transform:
            state = self.transform(state)
            action = self.transform(action)

        return state, action


def create_data_loader(data_dir, batch_size=64, shuffle=True, num_workers=4):
    """
    创建 DataLoader，用于加载行为克隆训练数据。
    :param data_dir: 存放数据的目录
    :param batch_size: 批量大小
    :param shuffle: 是否随机打乱数据
    :param num_workers: 数据加载的工作线程数
    :return: DataLoader 实例
    """
    # 创建数据集实例
    dataset = BehaviorCloningDataset(data_dir)

    # 创建 DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader
