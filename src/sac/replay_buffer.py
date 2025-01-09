import torch
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    用于存储智能体经验的Replay Buffer。
    """
    def __init__(self, buffer_size, state_dim, action_dim, batch_size=64):
        """
        :param buffer_size: 缓冲区的最大大小
        :param state_dim: 状态的维度
        :param action_dim: 动作的维度
        :param batch_size: 每次采样的批次大小
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer = deque(maxlen=buffer_size)
        self.pos = 0  # 当前的插入位置

    def push(self, state, action, reward, next_state, done):
        """
        将一条经验样本添加到回放缓冲区中。
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_state: 下一个状态
        :param done: 是否为终止状态
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        """
        从回放缓冲区中随机抽取一个批次的经验。
        :return: 一批经验，包含状态、动作、奖励、下一个状态和是否结束的标志
        """
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将各个部分转换成Tensor并返回
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def can_sample(self):
        """
        检查缓冲区中是否有足够的经验可以进行采样。
        :return: 是否可以采样（即缓冲区中是否有足够的经验）
        """
        return len(self.buffer) >= self.batch_size

    def size(self):
        """
        获取回放缓冲区中当前的经验数量。
        :return: 当前缓冲区大小
        """
        return len(self.buffer)

    def clear(self):
        """
        清空缓冲区。
        """
        self.buffer.clear()
