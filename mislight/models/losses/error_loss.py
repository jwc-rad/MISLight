from torch import nn
import torch.nn.functional as F
from .utils import softmax_helper, average_helper
from .generic_loss import GenericLoss

class MSELoss(GenericLoss):
    default_nonlin = [softmax_helper, softmax_helper]
    
    def __init__(self, apply_nonlin=[True, False], pre_avg=True, **kwargs):
        if not isinstance(apply_nonlin, (list, tuple)):
            apply_nonlin = [apply_nonlin,]*2
        generic_kwargs = dict(
            apply_nonlin=[n if a else None for n, a in zip(self.default_nonlin, apply_nonlin)],
            pre_avg=pre_avg
        )
        super().__init__(**generic_kwargs)
        
    def calc_loss(self, x, y):
        return F.mse_loss(x, y)
