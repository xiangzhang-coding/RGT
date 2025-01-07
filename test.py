# -*- coding: utf-8 -*-

"""
PyCharm -> RGT -> test
Author: XiangZhang
Date: 2025/1/5
"""

def main():
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np

    # 创建 TensorBoard 日志记录器
    writer = SummaryWriter(log_dir='test_logs')

    # 模拟训练过程
    for epoch in range(100):
        # 模拟损失值和准确率
        train_loss = 1 / (epoch + 1)
        val_accuracy = 1 - 1 / (epoch + 1)

        # 记录训练损失
        writer.add_scalar('Loss/train', train_loss, epoch)

        # 记录验证准确率
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    # 关闭日志记录器
    writer.close()
    return None


if __name__ == "__main__":
    main()
