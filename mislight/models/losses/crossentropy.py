import functools
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .utils import softmax_helper, log_softmax_helper, average_helper
from .generic_loss import GenericLoss

class CrossEntropyLoss(GenericLoss):
    default_nonlin = [softmax_helper, softmax_helper]
    
    def __init__(self, ce_kwargs, apply_nonlin=[False, False], pre_avg=True, argmax=False, **kwargs):
        if not isinstance(apply_nonlin, (list, tuple)):
            apply_nonlin = [apply_nonlin,]*2
        assert apply_nonlin[0]==False # nn.CrossEntropyLoss expects inputs as logits only
        generic_kwargs = dict(
            apply_nonlin=[n if a else None for n, a in zip(self.default_nonlin, apply_nonlin)],
            pre_avg=pre_avg
        )
        super().__init__(**generic_kwargs)
        
        self.argmax = argmax
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
    
    def calc_loss(self, x, y):
        if len(y.shape) == len(x.shape):
            if y.shape[1] > 1:
                if self.argmax:
                    y = torch.argmax(y, 1)
            else:
                y = y[:,0].long()
        else:
            y = y.long()
        return self.ce(x, y)
    
class KLDivLoss(GenericLoss):
    '''
    torch.nn.KLDivLoss expects the input tensor to be log probabilites by default
    torch.nn.KLDivLoss is pointwise KLDiv, so it should be BATCHMEAN / (x*y*z) to match behavior of CrossEntropyLoss
    '''
    default_nonlin = [log_softmax_helper, softmax_helper]
    
    def __init__(self, apply_nonlin=[True, True], pre_avg=True, **kwargs):
        if not isinstance(apply_nonlin, (list, tuple)):
            apply_nonlin = [apply_nonlin,]*2
        generic_kwargs = dict(
            apply_nonlin=[n if a else None for n, a in zip(self.default_nonlin, apply_nonlin)],
            pre_avg=pre_avg
        )        
        super().__init__(**generic_kwargs)

    def calc_loss(self, x, y):
        size = x.shape[2:].numel()
        return F.kl_div(x, y, reduction='batchmean') / size
        
        
# Copied from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/crossentropy.py
# Only called in nnUNet's losses. Not actually used in my losses.
class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())
