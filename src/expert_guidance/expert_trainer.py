import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from expert_sac import ExpertSAC
from expert_data_loader import ExpertDataLoader
from collections import deque
from torch.utils.data import DataLoader


class ExpertTrainer:
    def __init__(self, state_dim, action_dim, expert_data_path, batch_size=64, gamma=0.99, tau=0.005, alpha=0.2,
                 learning_rate=3e-4, num_epochs=100, imitation_learning_weight=0.1, use_imitation_learning=True):
        """
        ExpertTrainer 初始化
        :param state_dim: 状态空间的维度
        :param action_dim: 动作空间的维度
        :param expert_data_path: 专家数据路径
        :param batch_size: 每批的样本数
        :param gamma: 折扣因子
        :param tau: 目标网络软更新参数
        :param alpha: SAC的熵调节参数
        :param learning_rate: 学习率
        :param num_epochs: 训练轮数
        :param imitation_learning_weight: 模仿学习损失的权重
        :param use_imitation_learning: 是否使用模仿学习
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.imitation_learning_weight = imitation_learning_weight
        self.use_imitation_learning = use_imitation_learning

        # 加载专家数据
        self.expert_loader = ExpertDataLoader(data_path=expert_data_path, batch_size=batch_size)
        self.expert_dataloader = self.expert_loader.get_data_loader()

        # 初始化ExpertSAC训练器
        self.expert_sac = ExpertSAC(state_dim, action_dim, expert_data_path, batch_size=batch_size, gamma=gamma,
                                    tau=tau, alpha=alpha, learning_rate=learning_rate,
                                    use_imitation_learning=use_imitation_learning)

        # 用于存储训练过程的奖励
        self.episode_rewards = deque(maxlen=100)

    def train(self):
        """
        训练过程
        """
        # 记录训练过程中的平均奖励
        for epoch in range(self.num_epochs):
            episode_reward = 0.0

            for state, action, reward, next_state in self.expert_dataloader:
                state = state.to(torch.float32)
                action = action.to(torch.float32)
                next_state = next_state.to(torch.float32)
                reward = reward.to(torch.float32)

                # 在SAC中进行训练
                sac_loss = self.expert_sac.sac.update(state, action, reward, next_state)

                # 如果启用行为克隆，则计算模仿学习的损失并加权
                if self.use_imitation_learning:
                    imitation_loss = self.expert_sac._imitation_learning_loss(state, action)
                    total_loss = sac_loss + self.imitation_learning_weight * imitation_loss
                else:
                    total_loss = sac_loss

                # 更新SAC参数
                self.expert_sac.sac.optimizer.zero_grad()
                total_loss.backward()
                self.expert_sac.sac.optimizer.step()

                episode_reward += reward.mean().item()

            # 计算和记录平均奖励
            self.episode_rewards.append(episode_reward)
            print(f'Epoch {epoch + 1}/{self.num_epochs}, Reward: {episode_reward}')

        return self.episode_rewards

    def evaluate(self, eval_episodes=10):
        """
        评估模型
        :param eval_episodes: 评估的轮次
        :return: 平均评估奖励
        """
        total_reward = 0.0
        for _ in range(eval_episodes):
            state = self.expert_loader.reset()  # 假设 expert_loader 提供了 reset 方法来重置环境
            done = False
            while not done:
                action = self.expert_sac.sac.select_action(state)  # 使用 SAC 策略选择动作
                next_state, reward, done = self.expert_loader.step(action)  # 假设 expert_loader 提供了 step 方法
                state = next_state
                total_reward += reward

        avg_reward = total_reward / eval_episodes
        print(f'Average Evaluation Reward: {avg_reward}')
        return avg_reward

    def save(self, filepath):
        """
        保存模型
        :param filepath: 文件路径
        """
        self.expert_sac.save(filepath)

    def load(self, filepath):
        """
        加载模型
        :param filepath: 文件路径
        """
        self.expert_sac.load(filepath)


# 使用示例
if __name__ == "__main__":
    # 设置专家数据路径
    expert_data_path = "path_to_expert_data.json"

    # 创建并训练ExpertTrainer模型
    expert_trainer = ExpertTrainer(state_dim=3, action_dim=1, expert_data_path=expert_data_path, batch_size=128)
    rewards = expert_trainer.train()

    # 评估模型
    avg_reward = expert_trainer.evaluate(eval_episodes=10)

    # 保存模型
    expert_trainer.save('expert_sac_model')
