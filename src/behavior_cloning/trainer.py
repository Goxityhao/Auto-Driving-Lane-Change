import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import logging
from models.bc_model import BCModel
from src.behavior_cloning.data_loader import BehaviorCloningDataset
from utils import save_model, create_log_dir

# 设置日志记录
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


class BehaviorCloningTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建模型
        self.model = BCModel(input_dim=config['input_dim'], output_dim=config['output_dim']).to(self.device)

        # 定义优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = nn.MSELoss()  # 行为克隆通常使用均方误差损失

        # 创建数据加载器
        self.train_dataset = BehaviorCloningDataset(config['train_data_path'])
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['batch_size'], shuffle=True)

        # 创建日志和模型保存路径
        self.save_dir = create_log_dir(config['log_dir'])

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (states, actions) in enumerate(self.train_loader):
            states, actions = states.to(self.device), actions.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(states)

            # 计算损失
            loss = self.criterion(predictions, actions)
            total_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:  # 每100步打印一次日志
                logger.info(f"Epoch [{epoch}/{self.config['num_epochs']}], "
                            f"Step [{batch_idx}/{len(self.train_loader)}], "
                            f"Loss: {loss.item():.4f}")

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch [{epoch}/{self.config['num_epochs']}], "
                    f"Average Loss: {avg_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s")
        return avg_loss

    def train(self):
        for epoch in range(1, self.config['num_epochs'] + 1):
            avg_loss = self.train_epoch(epoch)

            # 每个epoch后保存模型
            if epoch % self.config['save_freq'] == 0:
                model_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
                save_model(self.model, model_path)
                logger.info(f"Model saved to {model_path}")

            # 如果训练损失低于某个阈值，提前停止训练
            if avg_loss < self.config['early_stopping_threshold']:
                logger.info(f"Early stopping at epoch {epoch}, loss reached {avg_loss:.4f}")
                break

    def evaluate(self, test_data_path):
        self.model.eval()
        test_dataset = BehaviorCloningDataset(test_data_path)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)

        total_loss = 0
        with torch.no_grad():
            for states, actions in test_loader:
                states, actions = states.to(self.device), actions.to(self.device)
                predictions = self.model(states)
                loss = self.criterion(predictions, actions)
                total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        logger.info(f"Test Loss: {avg_loss:.4f}")
        return avg_loss


# Example usage of the trainer
if __name__ == "__main__":
    config = {
        'input_dim': 10,  # 假设输入维度为10
        'output_dim': 3,  # 假设输出维度为3（例如：加速度、转向角、刹车力度）
        'train_data_path': 'data/train_data',  # 训练数据路径
        'batch_size': 64,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'save_freq': 10,  # 每10个epoch保存一次模型
        'log_dir': 'logs',  # 日志保存路径
        'early_stopping_threshold': 0.01,  # 提前停止的损失阈值
    }

    trainer = BehaviorCloningTrainer(config)
    trainer.train()
    trainer.evaluate('data/test_data')  # 使用测试数据评估模型
