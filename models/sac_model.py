import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json

# 读取配置文件
with open('./configs/sac_config.json', 'r') as f:
    config = json.load(f)


# Actor - 策略网络 (即生成动作的网络)
class Actor(nn.Module):
    """ 策略网络（Actor），生成动作。 """

    def __init__(self, state_dim, action_dim, action_range, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)  # 输出动作均值
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)  # 输出动作的标准差（log尺度）

        self.action_range = action_range  # 用于限制动作范围

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)  # 计算标准差

        return mean, std


# Critic - Q 网络（用于估计状态-动作值函数 Q(s, a)）
class Critic(nn.Module):
    """ Q网络，用于估计状态-动作值函数 Q(s, a)。 """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # 拼接状态和动作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# ValueNetwork - 用于估计状态值函数 V(s)
class ValueNetwork(nn.Module):
    """ Value 网络，用于估计 V(s) """

    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# SACModel
class SACModel:
    def __init__(self, action_range, config):
        # 从配置文件获取超参数
        self.gamma = config['training']['gamma']
        self.tau = config['training']['tau']
        self.alpha = config['training']['alpha']
        self.action_range = config['actor']['action_range']
        # 初始化各个网络
        self.actor = Actor(config["actor"]["state_dim"], config["actor"]["action_dim"], config["actor"]["action_range"],
                           config["actor"]["hidden_dim"])
        self.critic_1 = Critic(config["critic"]["state_dim"], config["critic"]["action_dim"],
                               config["critic"]["hidden_dim"])
        self.critic_2 = Critic(config["critic"]["state_dim"], config["critic"]["action_dim"],
                               config["critic"]["hidden_dim"])
        self.value_network = ValueNetwork(config["value_network"]["state_dim"], config["value_network"]["hidden_dim"])

        # 目标网络的拷贝
        self.target_critic_1 = Critic(config["critic"]["state_dim"], config["critic"]["action_dim"],
                                      config["critic"]["hidden_dim"])
        self.target_critic_2 = Critic(config["critic"]["state_dim"], config["critic"]["action_dim"],
                                      config["critic"]["hidden_dim"])
        self.target_value_network = ValueNetwork(config["value_network"]["state_dim"],
                                                 config["value_network"]["hidden_dim"])

        # 初始化目标网络参数
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_value_network.load_state_dict(self.value_network.state_dict())

        # 设置优化器
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=config['training']['critic_lr'])
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=config['training']['critic_lr'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['training']['actor_lr'])
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=config['training']['value_lr'])

    def select_action(self, state, deterministic=False):
        """
        根据当前状态选择动作。根据策略网络输出的均值和标准差来选择动作。
        :param state: 当前状态
        :param deterministic: 是否使用确定性动作（即不随机选择）
        :return: 选择的动作
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, std = self.actor(state)

        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()  # 通过重参数化采样

        # 将动作限制在给定的范围内
        action = torch.tanh(action) * self.action_range
        return action.squeeze(0).detach().numpy()

    def update(self, state, action, reward, next_state, done):
        """
        更新SAC模型的Q网络和策略网络。
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 当前奖励
        :param next_state: 下一个状态
        :param done: 是否结束
        :return: 损失值
        """
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(0)

        # 计算目标Q值（使用目标Q网络）
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_state)
            next_action = torch.tanh(next_action) * self.action_range
            q_target_1 = self.target_critic_1(next_state, next_action)
            q_target_2 = self.target_critic_2(next_state, next_action)
            v_target = self.target_value_network(next_state)
            q_target = torch.min(q_target_1, q_target_2) - self.alpha * next_log_prob

        # 计算Q网络损失
        q_value_1 = self.critic_1(state, action)
        q_value_2 = self.critic_2(state, action)

        q_loss_1 = F.mse_loss(q_value_1, reward + self.gamma * (1 - done) * q_target)
        q_loss_2 = F.mse_loss(q_value_2, reward + self.gamma * (1 - done) * q_target)

        # 更新Q网络
        self.critic_optimizer_1.zero_grad()
        q_loss_1.backward()
        self.critic_optimizer_1.step()

        self.critic_optimizer_2.zero_grad()
        q_loss_2.backward()
        self.critic_optimizer_2.step()

        # 策略网络更新
        action, log_prob = self.actor(state)
        action = torch.tanh(action) * self.action_range
        q_value_1 = self.critic_1(state, action)
        q_value_2 = self.critic_2(state, action)
        q_value = torch.min(q_value_1, q_value_2)

        policy_loss = (self.alpha * log_prob - q_value).mean()

        # 更新策略网络
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # 更新目标网络
        self._soft_update(self.target_critic_1, self.critic_1)
        self._soft_update(self.target_critic_2, self.critic_2)
        self._soft_update(self.target_value_network, self.value_network)

        return q_loss_1.item() + q_loss_2.item(), policy_loss.item()

    def _soft_update(self, target_net, net):
        """
        软更新目标网络参数。
        :param target_net: 目标网络
        :param net: 当前网络
        """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param)
