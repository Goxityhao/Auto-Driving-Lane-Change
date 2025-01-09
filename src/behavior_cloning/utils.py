import os
import torch
import logging
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import shutil


# 设置日志记录
def setup_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # 文件输出
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


# 创建日志文件夹
def create_log_dir(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# 保存模型
def save_model(model, file_path):
    """
    保存模型的状态字典到指定路径
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


# 加载模型
def load_model(model, file_path):
    """
    加载模型的状态字典
    """
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")
    else:
        print(f"Model file not found at {file_path}")
    return model


# 删除文件夹
def remove_dir(directory):
    """
    删除目录及其中所有内容
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Directory {directory} removed.")
    else:
        print(f"Directory {directory} not found.")


# 创建并返回TensorBoard的SummaryWriter对象
def get_tensorboard_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer


# 设置并返回设备（CPU或GPU）
def get_device():
    """
    获取训练时使用的设备，支持CUDA（GPU）和CPU
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


# 计算训练的平均损失
def compute_avg_loss(losses):
    """
    计算平均损失
    """
    return sum(losses) / len(losses)


# 计算训练时间（格式化为时:分:秒）
def format_time(seconds):
    """
    将训练时间转化为时:分:秒格式
    """
    mins = seconds // 60
    secs = seconds % 60
    hours = mins // 60
    mins = mins % 60
    return f"{int(hours):02}:{int(mins):02}:{int(secs):02}"


# 用于保存训练过程中的指标（如损失值、学习率等）到TensorBoard
def log_to_tensorboard(writer, epoch, loss, lr=None):
    """
    将训练过程中的损失和学习率等信息记录到TensorBoard
    """
    writer.add_scalar('Loss/train', loss, epoch)
    if lr:
        writer.add_scalar('Learning_Rate', lr, epoch)

