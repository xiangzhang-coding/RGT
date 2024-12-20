# 代码的核心目的是 从配置选项中动态创建损失函数实例。
#
# 输入: 一个包含损失函数配置信息的字典 opt。
# 输出: 根据配置创建的损失函数实例。
from copy import deepcopy

from basicsr.utils import get_root_logger

# LOSS_REGISTRY 是一个 注册表对象，用来动态注册和管理损失函数。
# 每个损失函数（如 L1Loss、MSELoss）都通过注册的方式添加到这个注册表中，便于通过字符串键值动态调用。
from basicsr.utils.registry import LOSS_REGISTRY
from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, WeightedTVLoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty)

# 用于限制模块中可以被 import * 导入的内容。只有 __all__ 列出的函数或类会被导入。
# 这里列出了该模块包含的一些预定义损失函数。
__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'GANLoss', 'gradient_penalty_loss',
    'r1_penalty', 'g_path_regularize'
]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt) # 使用 deepcopy 防止修改原始的 opt，因为后续会从字典中弹出（pop）某些键值。opt 是一个包含损失函数类型和相关参数的字典
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
