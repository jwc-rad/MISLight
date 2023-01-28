import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

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