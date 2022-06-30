'''Adapted from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/ & https://github.com/JunMa11/SegLoss/blob/master/test/nnUNetV2/loss_functions/
Targets shape: (b, c, x, y, z) or (b, x, y, z).
'''

import torch
from torch import nn
import numpy as np
from .dice_loss import DiceLoss
from .crossentropy import CrossEntropyLoss
from .focal_loss import GenericFocalLoss
from .utils import softmax_helper

class DiceCELoss(nn.Module):
    def __init__(self, dice_kwargs, dice_generic_kwargs, ce_kwargs, ce_generic_kwargs, weight_dice=1, weight_ce=1):
        super().__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dc = DiceLoss(dice_kwargs, **dice_generic_kwargs)
        self.ce = CrossEntropyLoss(ce_kwargs, **ce_generic_kwargs)

    def forward(self, x, y):
        dc_loss = self.dc(x, y) if self.weight_dice != 0 else 0
        ce_loss = self.ce(x, y) if self.weight_ce != 0 else 0
        result = self.weight_dice * dc_loss + self.weight_ce * ce_loss      
        return result
    
class DiceFocalLoss(nn.Module):
    def __init__(self, dice_kwargs, dice_generic_kwargs, ce_kwargs, ce_generic_kwargs, weight_dice=1, weight_ce=1):
        super().__init__()

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dc = DiceLoss(dice_kwargs, **dice_generic_kwargs)
        self.ce = GenericFocalLoss(ce_kwargs, **ce_generic_kwargs)

    def forward(self, x, y):
        dc_loss = self.dc(x, y) if self.weight_dice != 0 else 0
        ce_loss = self.ce(x, y) if self.weight_ce != 0 else 0
        result = self.weight_dice * dc_loss + self.weight_ce * ce_loss      
        return result
