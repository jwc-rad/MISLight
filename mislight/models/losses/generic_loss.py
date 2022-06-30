from abc import abstractmethod

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .utils import softmax_helper, average_helper

class GenericLoss(nn.Module):
    '''
    X must be (b, c, x, y(, z)) logits
    Y must be (b, c, x, y(, z)) logits or probabilities/one-hot OR (b, x, y(, z)) labels    
    Args:
        apply_nonlin: applying nonlinearity to X and Y
        pre_avg: For multiple targets, if True, average probabilities then calculate loss, if False, average losses.
    '''
    def __init__(self, apply_nonlin=[softmax_helper, None], pre_avg=True):
        super().__init__()
        assert isinstance(apply_nonlin, (list, tuple)) and len(apply_nonlin)==2
        self.apply_nonlin_x, self.apply_nonlin_y = apply_nonlin
        self.pre_avg = pre_avg
            
    @abstractmethod
    def calc_loss(self, x, y):
        pass
            
    def forward(self, x, y):   
        if self.apply_nonlin_x:
            x = self.apply_nonlin_x(x)
            
        if isinstance(y, (list, tuple)):
            if self.pre_avg:
                if self.apply_nonlin_y:
                    outs = []
                    for y1 in y:
                        y1 = self.apply_nonlin_y(y1)
                        outs.append(y1)
                    y = average_helper(outs)
                else:
                    y = average_helper(y)
                return self.calc_loss(x, y)
            else:
                outs = []
                for y1 in y:            
                    if self.apply_nonlin_y:
                        y1 = self.apply_nonlin_y(y1)
                    outs.append(self.calc_loss(x, y1))
                return average_helper(outs)
        else:
            if self.apply_nonlin_y:
                y = self.apply_nonlin_y(y)
            return self.calc_loss(x, y)
