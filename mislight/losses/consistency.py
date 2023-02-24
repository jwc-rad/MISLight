import math
from typing import Callable, Optional, Union
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction

class MSEConLoss(_Loss):
    """
    input, target are both logits with shapes BNHW[D] where N is number of classes
    """
    def __init__(
        self,
        include_background: bool = True,
        sigmoid: bool = False,
        softmax: bool = False,
        temperature_y: float = 1,
        reduction: str = 'mean',
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            temperature_y: soften(t > 1) or sharpen(t < 1) predictions.
        """
        super().__init__(reduction=reduction)
        if int(sigmoid) + int(softmax) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True].")
        self.include_background = include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        assert temperature_y > 0
        self.temperature_y = temperature_y
            
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target / self.temperature_y
        
        if self.sigmoid:
            input = torch.sigmoid(input)
            target = torch.sigmoid(target)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1) 
                target = torch.softmax(target, 1) 
        
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]
                
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
            
        return F.mse_loss(input, target, reduction=self.reduction)
    
class DiceConLoss(_Loss):
    """
    Modified from monai.losses.DiceLoss.
    input, target are both logits with shapes BNHW[D] where N is number of classes
    """
    def __init__(
        self,
        include_background: bool = True,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = 'mean',
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
    ) -> None:
        super().__init__(reduction=reduction)
        if other_act is not None and not callable(other_act):
            raise TypeError(f"other_act must be None or callable but is {type(other_act).__name__}.")
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None].")
        self.include_background = include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)
            target = torch.sigmoid(target) > 0.5

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)
                target = target.argmax(1, keepdim=True)
                target = one_hot(target, num_classes=n_pred_ch)

        if self.other_act is not None:
            input = self.other_act(input)
            target = self.other_act(target) > 0.5

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f
    
    
class KLDivConLoss(_Loss):
    """
    input, target are both logits with shapes BNHW[D] where N is number of classes
    torch.nn.KLDivLoss expects the input tensor to be log probabilites by default
    torch.nn.KLDivLoss is pointwise KLDiv, so it should be BATCHMEAN / (x*y*z) to match behavior of CrossEntropyLoss
    """
    def __init__(
        self,
        temperature: float = 1,
        reduction: str = 'default',
    ) -> None:
        super().__init__(reduction=reduction)
        assert temperature > 0
        self.temperature = temperature
            
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input / self.temperature
        target = target / self.temperature
        
        input = torch.log_softmax(input, 1)
        target = torch.softmax(target, 1)
        
        if self.reduction == 'default':
            size = math.prod(input.shape[2:])
            loss = F.kl_div(input, target, reduction='batchmean') / size
        else:
            loss = F.kl_div(input, target, reduction=self.reduction)
        return loss * (self.temperature ** 2)