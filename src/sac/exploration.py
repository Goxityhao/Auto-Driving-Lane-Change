import torch
import torch.nn.functional as F
import numpy as np


class GaussianNoiseExploration:
    """
    使用高斯噪声来进行探索性动作选择。
    这将会在原始动作选择上加上噪声，推动探索。
    """

    def __init__(self, action_dim, noise_std=0.1):
        """
        :param action_dim: 动作空间的维度
        :param noise_std: 噪声的标准差
        """
        self.action_dim = action_dim
        self.noise_std = noise_std

    def add_noise(self, action):
        """
        给动作添加高斯噪声，用于增强探索性。
        :param action: 来自策略网络的动作（可以是连续动作）
        :return: 加噪声后的动作
        """
        noise = torch.randn_like(action) * self.noise_std
        noisy_action = action + noise
        return noisy_action


class RandomExploration:
    """
    使用完全随机的动作选择进行探索。
    这种策略将动作完全随机化，有助于在初期阶段进行广泛的探索。
    """

    def __init__(self, action_range):
        """
        :param action_range: 动作空间的范围，用于限制生成的动作
        """
        self.action_range = action_range

    def select_action(self):
        """
        随机选择一个动作。
        :return: 随机选择的动作
        """
        action = np.random.uniform(-self.action_range, self.action_range, size=self.action_range)
        return torch.tensor(action, dtype=torch.float32)


class SACExploration:
    """
    SAC 探索策略，利用熵来保持动作的多样性。
    在SAC中，探索通常是通过策略网络的输出实现的。
    """

    def __init__(self, policy_network, action_range, alpha=0.2):
        """
        :param policy_network: 策略网络
        :param action_range: 动作的范围，用于对动作进行裁剪
        :param alpha: 熵项的系数
        """
        self.policy_network = policy_network
        self.action_range = action_range
        self.alpha = alpha

    def select_action(self, state, deterministic=False):
        """
        使用策略网络选择动作，并根据需要进行熵正则化。
        :param state: 当前状态
        :param deterministic: 是否选择确定性动作（即不添加噪声）
        :return: 选择的动作
        """
        mean, std = self.policy_network(state)

        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()  # 使用重参数化技巧进行采样

        # 对动作进行裁剪，以确保它们在合法范围内
        action = torch.tanh(action) * self.action_range
        return action


class ExplorationStrategy:
    """
    综合探索策略，允许多种探索策略的组合。
    """

    def __init__(self, strategy_type='gaussian', action_dim=None, action_range=None, noise_std=0.1, policy_network=None,
                 alpha=0.2):
        """
        :param strategy_type: 探索策略类型 ('gaussian', 'random', 'sac')
        :param action_dim: 动作空间的维度
        :param action_range: 动作范围
        :param noise_std: 高斯噪声的标准差（适用于 'gaussian'）
        :param policy_network: 策略网络（适用于 'sac'）
        :param alpha: 熵正则化系数（适用于 'sac'）
        """
        if strategy_type == 'gaussian':
            self.exploration = GaussianNoiseExploration(action_dim, noise_std)
        elif strategy_type == 'random':
            self.exploration = RandomExploration(action_range)
        elif strategy_type == 'sac':
            self.exploration = SACExploration(policy_network, action_range, alpha)
        else:
            raise ValueError("Unsupported exploration strategy type")

    def select_action(self, state, deterministic=False):
        """
        根据所选择的探索策略选择动作。
        :param state: 当前状态
        :param deterministic: 是否选择确定性动作
        :return: 选择的动作
        """
        return self.exploration.select_action(state, deterministic)
