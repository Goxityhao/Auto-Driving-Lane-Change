import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from models.sac_model import Actor, Critic, QValueNetwork, ValueNetwork
from utils import save_model, get_device
import torch.nn.functional as F


class SACAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = get_device()

        # 初始化超参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config['gamma']  # 折扣因子
        self.tau = config['tau']  # 软更新的目标网络的参数
        self.alpha = config['alpha']  # 温度参数（控制探索/利用之间的平衡）
        self.lr_actor = config['lr_actor']
        self.lr_critic = config['lr_critic']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']

        # 创建策略网络 (Actor) 和 Q 网络 (Critic)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.value_network = ValueNetwork(state_dim).to(self.device)

        # 创建目标网络 (Target Networks)
        self.target_critic_1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(self.device)
        self.target_value_network = ValueNetwork(state_dim).to(self.device)

        # 初始化目标网络的参数
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_value_network.load_state_dict(self.value_network.state_dict())

        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.lr_critic)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.lr_critic)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr_critic)

        # 初始化经验回放缓冲区
        self.replay_buffer = deque(maxlen=self.buffer_size)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = self.actor(state)
        action = action.detach().cpu().numpy()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self):
        # 如果经验回放中样本不足，则跳过更新
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从经验回放中采样一批数据
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 计算目标 Q 值
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_q1 = self.target_critic_1(next_states, next_actions)
            next_q2 = self.target_critic_2(next_states, next_actions)
            next_value = self.target_value_network(next_states)

            target_q = rewards + (1 - dones) * self.gamma * (torch.min(next_q1, next_q2) - self.alpha * next_value)

        # 更新 Critic 网络
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_1_loss = F.mse_loss(q1, target_q)
        critic_2_loss = F.mse_loss(q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新 Value 网络
        value = self.value_network(states)
        value_loss = F.mse_loss(value, torch.min(q1, q2) - self.alpha * next_value)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 更新 Actor 网络
        actions_pred = self.actor(states)
        q1 = self.critic_1(states, actions_pred)
        q2 = self.critic_2(states, actions_pred)
        actor_loss = (self.alpha * self.value_network(states) - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.target_critic_1, self.critic_1)
        self.soft_update(self.target_critic_2, self.critic_2)
        self.soft_update(self.target_value_network, self.value_network)

    def soft_update(self, target_network, source_network):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )

    def save(self, model_path):
        save_model(self.actor, model_path + "_actor.pth")
        save_model(self.critic_1, model_path + "_critic_1.pth")
        save_model(self.critic_2, model_path + "_critic_2.pth")
        save_model(self.value_network, model_path + "_value_network.pth")

    def load(self, model_path):
        self.actor = load_model(self.actor, model_path + "_actor.pth")
        self.critic_1 = load_model(self.critic_1, model_path + "_critic_1.pth")
        self.critic_2 = load_model(self.critic_2, model_path + "_critic_2.pth")
        self.value_network = load_model(self.value_network, model_path + "_value_network.pth")
