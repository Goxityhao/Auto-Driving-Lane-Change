import torch
import numpy as np
import keyboard  # 用于检测键盘按键
from models.sac_model import SACModel
from src.sac.replay_buffer import ReplayBuffer
from utils import save_model, get_device
import random

class SACTrainer:
    """
    用于训练SAC模型，并结合在线专家指导。
    """

    def __init__(self, state_dim, action_dim, action_range, buffer_size=1000000, batch_size=64, gamma=0.99, tau=0.005, alpha=0.2):
        """
        :param state_dim: 状态维度
        :param action_dim: 动作维度
        :param action_range: 动作范围
        :param buffer_size: 回放缓冲区大小
        :param batch_size: 每次训练的批量大小
        :param gamma: 折扣因子
        :param tau: 目标网络更新速率
        :param alpha: 熵系数
        """
        self.device = get_device()
        self.model = SACModel(state_dim, action_dim, action_range, gamma, tau, alpha).to(self.device)
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, batch_size)
        self.batch_size = batch_size

    def train(self, env, max_episodes=1000, max_timesteps_per_episode=1000, save_interval=100):
        """
        训练SAC模型，并结合专家的指导。
        :param env: 环境
        :param max_episodes: 最大训练回合数
        :param max_timesteps_per_episode: 每回合最大步数
        :param save_interval: 保存模型的间隔
        """
        total_timesteps = 0
        for episode in range(max_episodes):
            state = env.reset()
            episode_reward = 0

            for t in range(max_timesteps_per_episode):
                total_timesteps += 1

                # 判断是否使用专家控制
                expert_action = None
                if self.need_expert_guidance():
                    # 由专家提供动作
                    expert_action = self.expert_control(state)

                # 如果专家没有控制，则使用模型的动作
                if expert_action is None:
                    action = self.model.select_action(state)
                else:
                    action = expert_action

                # 在环境中执行动作
                next_state, reward, done, info = env.step(action)

                # 判断是否发生碰撞
                if 'collision' in info and info['collision']:
                    print("Collision detected! Terminating this episode and moving to the next one.")
                    break  # 碰撞发生，终止当前回合

                # 存储专家经验
                if expert_action is not None:
                    self.replay_buffer.push(state, expert_action, reward, next_state, done)

                # 存储智能体经验
                if expert_action is None:
                    self.replay_buffer.push(state, action, reward, next_state, done)

                # 更新模型
                if self.replay_buffer.can_sample():
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample()
                    loss_q, loss_policy = self.model.update(states, actions, rewards, next_states, dones)

                # 打印每个回合的奖励
                episode_reward += reward

                if done:
                    break

            # 打印训练进度
            print(f"Episode {episode+1}/{max_episodes}, Reward: {episode_reward}")

            # 保存模型
            if (episode + 1) % save_interval == 0:
                save_model(self.model, f"sac_model_episode_{episode+1}.pth")

    def need_expert_guidance(self):
        """
        判断是否需要专家干预。通过键盘输入控制。
        :return: 是否需要专家控制
        """
        # 如果按下“w”，“s”，“a”，“d”任意键，启用专家接管
        return keyboard.is_pressed('w') or keyboard.is_pressed('s') or keyboard.is_pressed('a') or keyboard.is_pressed('d')

    def expert_control(self, state):
        """
        根据键盘输入生成专家控制动作。
        :param state: 当前状态
        :return: 专家生成的动作
        """
        # 根据键盘输入控制专家的行为
        if keyboard.is_pressed('w'):
            # 加速
            action = [1.0, 0.0]  # 假设动作是加速
        elif keyboard.is_pressed('s'):
            # 减速
            action = [-1.0, 0.0]  # 假设动作是减速
        elif keyboard.is_pressed('a'):
            # 左转
            action = [0.0, -1.0]  # 假设动作是左转
        elif keyboard.is_pressed('d'):
            # 右转
            action = [0.0, 1.0]  # 假设动作是右转
        else:
            action = None

        return action

# 使用示例
if __name__ == "__main__":
    from src.simulation.carla_env import CarlaEnv  # 请根据实际环境导入
    env = CarlaEnv()

    # 初始化Trainer
    trainer = SACTrainer(state_dim=3, action_dim=2, action_range=1)

    # 开始训练
    trainer.train(env, max_episodes=1000, max_timesteps_per_episode=1000)
