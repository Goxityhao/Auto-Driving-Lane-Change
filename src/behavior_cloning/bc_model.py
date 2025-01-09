import torch
import torch.nn as nn
import torch.nn.functional as F
import json


# 加载配置文件
def load_config(config_path='./configs/behavior_cloning_config.json'):
    with open(config_path, 'r') as file:
        return json.load(file)


# 定义行为克隆模型
class BCModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units):
        """
        初始化行为克隆模型。
        :param state_dim: 输入状态空间的维度（包括车辆位置、速度、周围车辆信息等）。
        :param action_dim: 输出的动作维度，通常包括加速度/刹车和转向角度。
        :param hidden_units: 每个全连接层的隐藏单元数量。
        """
        super(BCModel, self).__init__()

        # 假设输入为一维的状态向量而不是图像
        # 全连接层，处理输入状态特征
        self.fc1 = nn.Linear(state_dim, hidden_units[0])  # 第一个全连接层
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])  # 第二个全连接层
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])  # 第三个全连接层

        # 动作输出层，分别输出加速度/刹车控制和转向角度
        self.fc4 = nn.Linear(hidden_units[2], action_dim)  # 输出层，action_dim=2（加速度、转向角度）

    def forward(self, state):
        """
        前向传播，执行状态输入到动作输出的映射。
        :param state: 输入状态数据，大小为(batch_size, state_dim)。
        :return: 预测的动作（加速度/刹车控制、转向角度）。
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # 最终的动作输出
        action = self.fc4(x)

        return action


# 示例：如何创建和测试BC模型
if __name__ == "__main__":
    # 加载配置文件
    config = load_config('./behavior_cloning_config.json')

    # 从配置文件中提取超参数
    state_dim = len(config['data']['image_size'])  # 假设state_dim基于图像尺寸（例如宽高），可以修改为根据实际的状态维度
    action_dim = config['model']['num_classes']  # 输出动作维度
    hidden_units = [layer['units'] for layer in config['model']['dense_layers']]  # 提取隐藏层的单元数

    # 创建模型
    model = BCModel(state_dim=state_dim, action_dim=action_dim, hidden_units=hidden_units)

    # 测试模型输出
    dummy_state = torch.randn(1, state_dim)  # 假设状态空间有8个特征
    output = model(dummy_state)
    print(output)
