import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def average_helper(xs):
    if isinstance(xs, (list, tuple)):
        out = xs[0]
        for x in xs[1:]:
            out += x
        out = out / len(xs)
        return out
    else:
        return xs

def softmax_helper(x):
    return F.softmax(x, dim=1)

def log_softmax_helper(x):
    return F.log_softmax(x, dim=1)

def argmax_helper(x):
    return torch.argmax(x, 1)

def softmax_helper_temperature(use_argmax):
    return lambda x, t=1: _softmax_helper_temperature(x, t, use_argmax)
    
def _softmax_helper_temperature(x, t=1, use_argmax=0):
    if t <= use_argmax:
        return torch.argmax(x, 1)
    else:
        return F.softmax(x/t, dim=1)
    
def log_softmax_helper_temperature(x, t=1):
    return F.log_softmax(x/t, dim=1)

# Copied from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

# Copied from: https://github.com/HiLab-git/SSL4MIS/blob/master/code/utils/ramps.py
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_rampup_weight(t, t_max):
    return sigmoid_rampup(t, t_max)
