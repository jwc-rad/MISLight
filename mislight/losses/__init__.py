import re
import torch
import torch.nn as nn
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, TverskyLoss #, DiceFocalLoss

def define_loss(loss_name):
    if loss_name.lower() == 'ce':
        loss = nn.CrossEntropyLoss()
    elif loss_name.lower() == 'focal':
        loss = FocalLoss(include_background=False, gamma=2, weight=0.5)
    elif loss_name.lower() == 'dice':
        loss = DiceLoss(softmax=True, include_background=False)
    elif loss_name.lower() == 'dicece':
        loss = DiceCELoss(softmax=True, include_background=False)
    elif loss_name.lower() == 'tversky':
        loss = TverskyLoss(softmax=True, include_background=False)
    else:
        raise NotImplementedError(f'loss name [{loss_name}] is not recognized')        
    return loss