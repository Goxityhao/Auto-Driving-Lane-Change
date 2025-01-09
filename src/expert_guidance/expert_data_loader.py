import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ExpertDataset(Dataset):
    """ 自定义数据集，用于加载专家数据 """

    def __init__(self, data_path, normalize=False):
        """
        初始化函数
        :param data_path: 存放专家数据的路径，可能是JSON文件、CSV文件或HDF5文件
        :param normalize: 是否进行归一化处理
        """
        self.data_path = data_path
        self.normalize = normalize
        self.expert_data = self._load_data(data_path)
        self.state_dim = self.expert_data['states'].shape[1]  # 状态的维度
        self.action_dim = self.expert_data['actions'].shape[1]  # 动作的维度

    def _load_data(self, data_path):
        """
        加载专家数据（假设数据是JSON格式的）
        :param data_path: 存放专家数据的路径
        :return: 字典格式的数据，包括状态、动作和奖励等
        """
        # 假设数据是JSON格式
        with open(data_path, 'r') as f:
            data = json.load(f)

        # 从加载的数据中提取状态、动作和奖励
        states = np.array(data['states'], dtype=np.float32)
        actions = np.array(data['actions'], dtype=np.float32)
        rewards = np.array(data['rewards'], dtype=np.float32)
        next_states = np.array(data['next_states'], dtype=np.float32)

        # 进行归一化（如果需要）
        if self.normalize:
            states = self._normalize(states)
            next_states = self._normalize(next_states)

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states
        }

    def _normalize(self, data):
        """
        对数据进行归一化处理
        :param data: 输入数据
        :return: 归一化后的数据
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)

    def __len__(self):
        """ 返回数据集的大小 """
        return len(self.expert_data['states'])

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        :param idx: 索引
        :return: 样本数据（状态、动作、奖励、下一个状态）
        """
        state = torch.tensor(self.expert_data['states'][idx], dtype=torch.float32)
        action = torch.tensor(self.expert_data['actions'][idx], dtype=torch.float32)
        reward = torch.tensor(self.expert_data['rewards'][idx], dtype=torch.float32)
        next_state = torch.tensor(self.expert_data['next_states'][idx], dtype=torch.float32)

        return state, action, reward, next_state


class ExpertDataLoader:
    """ 加载专家数据的工具类 """

    def __init__(self, data_path, batch_size=64, normalize=False, shuffle=True, num_workers=4):
        """
        初始化函数
        :param data_path: 存放专家数据的路径
        :param batch_size: 每个批次的样本数
        :param normalize: 是否对数据进行归一化
        :param shuffle: 是否打乱数据
        :param num_workers: 用于加载数据的线程数
        """
        self.batch_size = batch_size
        self.normalize = normalize
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset = ExpertDataset(data_path, normalize)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

    def get_data_loader(self):
        """
        返回用于训练的 DataLoader 对象
        :return: DataLoader
        """
        return self.dataloader

    def get_state_dim(self):
        """ 返回状态的维度 """
        return self.dataset.state_dim

    def get_action_dim(self):
        """ 返回动作的维度 """
        return self.dataset.action_dim


# 使用示例
if __name__ == "__main__":
    # 设置专家数据路径
    expert_data_path = "path_to_expert_data.json"

    # 创建专家数据加载器
    expert_loader = ExpertDataLoader(data_path=expert_data_path, batch_size=128, normalize=True)

    # 获取数据加载器
    dataloader = expert_loader.get_data_loader()

    # 示例：遍历数据集并打印状态、动作和奖励
    for state, action, reward, next_state in dataloader:
        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
