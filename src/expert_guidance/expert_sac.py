import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from models.sac_model import SACModel # 假设你已经实现了SAC算法
from expert_data_loader import ExpertDataLoader
from torch.utils.data import DataLoader


class ExpertSAC:
    def __init__(self, state_dim, action_dim, expert_data_path, batch_size=64, gamma=0.99, tau=0.005, alpha=0.2,
                 learning_rate=3e-4, normalize_expert_data=False, use_imitation_learning=True):
        """
        ExpertSAC 初始化
        :param state_dim: 状态空间的维度
        :param action_dim: 动作空间的维度
        :param expert_data_path: 专家数据路径
        :param batch_size: 每批的样本数
        :param gamma: 折扣因子
        :param tau: 目标网络软更新参数
        :param alpha: SAC的熵调节参数
        :param learning_rate: 学习率
        :param normalize_expert_data: 是否归一化专家数据
        :param use_imitation_learning: 是否使用行为克隆进行模仿学习
        """
        # 设置参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.normalize_expert_data = normalize_expert_data
        self.use_imitation_learning = use_imitation_learning

        # 初始化SAC算法
        self.sac = SACModel(state_dim, action_dim, gamma, tau, alpha, learning_rate)

        # 加载专家数据
        self.expert_loader = ExpertDataLoader(data_path=expert_data_path, batch_size=batch_size,
                                              normalize=normalize_expert_data)
        self.expert_dataloader = self.expert_loader.get_data_loader()

    def train(self, num_epochs=100, imitation_learning_weight=0.1):
        """
        训练函数
        :param num_epochs: 训练的轮数
        :param imitation_learning_weight: 模仿学习的损失权重
        """
        # 用于存储训练过程中的损失
        episode_rewards = []

        # 训练开始
        for epoch in range(num_epochs):
            episode_reward = 0.0
            for state, action, reward, next_state in self.expert_dataloader:
                state = state.to(torch.float32)
                action = action.to(torch.float32)
                next_state = next_state.to(torch.float32)
                reward = reward.to(torch.float32)

                # 在SAC中进行训练
                sac_loss = self.sac.update(state, action, reward, next_state)

                # 如果启用行为克隆，则进行模仿学习损失计算
                if self.use_imitation_learning:
                    imitation_loss = self._imitation_learning_loss(state, action)
                    total_loss = sac_loss + imitation_learning_weight * imitation_loss
                else:
                    total_loss = sac_loss

                # 更新SAC参数
                self.sac.optimizer.zero_grad()
                total_loss.backward()
                self.sac.optimizer.step()

                episode_reward += reward.mean().item()

            episode_rewards.append(episode_reward)
            print(f'Epoch {epoch + 1}/{num_epochs}, Reward: {episode_reward}')

        return episode_rewards

    def _imitation_learning_loss(self, states, actions):
        """
        计算行为克隆的损失，行为克隆通过最小化专家动作与当前策略动作之间的差异来训练。
        :param states: 当前状态
        :param actions: 当前动作
        :return: 行为克隆损失
        """
        # 使用SAC的当前策略预测动作
        predicted_actions, _ = self.sac.policy(states)

        # 行为克隆损失：最小化当前动作和专家数据中动作之间的均方误差
        imitation_loss = torch.mean((predicted_actions - actions) ** 2)
        return imitation_loss

    def save(self, filepath):
        """
        保存训练的SAC模型
        :param filepath: 保存路径
        """
        torch.save(self.sac.policy.state_dict(), filepath + '_policy.pth')
        torch.save(self.sac.q_network.state_dict(), filepath + '_q_network.pth')
        torch.save(self.sac.target_q_network.state_dict(), filepath + '_target_q_network.pth')

    def load(self, filepath):
        """
        加载训练好的SAC模型
        :param filepath: 模型文件路径
        """
        self.sac.policy.load_state_dict(torch.load(filepath + '_policy.pth'))
        self.sac.q_network.load_state_dict(torch.load(filepath + '_q_network.pth'))
        self.sac.target_q_network.load_state_dict(torch.load(filepath + '_target_q_network.pth'))


# 使用示例
if __name__ == "__main__":
    # 设置专家数据路径
    expert_data_path = "path_to_expert_data.json"

    # 创建并训练ExpertSAC模型
    expert_sac = ExpertSAC(state_dim=3, action_dim=1, expert_data_path=expert_data_path, batch_size=128)
    rewards = expert_sac.train(num_epochs=100, imitation_learning_weight=0.1)

    # 保存模型
    expert_sac.save('sac_expert_model')
