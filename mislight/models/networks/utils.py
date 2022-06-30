import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

def init_weights(net, init_type='xavier'):
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            if init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return net

# Adapted from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_modular_UNet.py
def get_default_network_config(dim=3, dropout_p=None, nonlin="LeakyReLU", norm_type="in"):
    props = {}
    if dim == 2:
        props['conv_op_base'] = nn.Conv2d
        props['conv_op'] = nn.Conv2d
        props['dropout_op'] = nn.Dropout2d
    elif dim == 3:
        props['conv_op_base'] = nn.Conv3d
        props['conv_op'] = nn.Conv3d
        props['dropout_op'] = nn.Dropout3d
    else:
        raise NotImplementedError

    if norm_type == "bn":
        if dim == 2:
            props['norm_op'] = nn.BatchNorm2d
        elif dim == 3:
            props['norm_op'] = nn.BatchNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    elif norm_type == "in":
        if dim == 2:
            props['norm_op'] = nn.InstanceNorm2d
        elif dim == 3:
            props['norm_op'] = nn.InstanceNorm3d
        props['norm_op_kwargs'] = {'eps': 1e-5, 'affine': True}
    else:
        raise NotImplementedError

    if dropout_p is None:
        props['dropout_op'] = None
        props['dropout_op_kwargs'] = {'p': 0, 'inplace': True}
    else:
        props['dropout_op_kwargs'] = {'p': dropout_p, 'inplace': True}

    props['conv_op_kwargs'] = {'stride': 1, 'dilation': 1, 'bias': True}  # kernel size will be set by network!

    if nonlin == "LeakyReLU":
        props['nonlin'] = nn.LeakyReLU
        props['nonlin_kwargs'] = {'negative_slope': 1e-2, 'inplace': True}
    elif nonlin == "ReLU":
        props['nonlin'] = nn.ReLU
        props['nonlin_kwargs'] = {'inplace': True}
    elif nonlin == "PReLU":
        props['nonlin'] = nn.PReLU  
        props['nonlin_kwargs'] = {'num_parameters': 1} # init is done separately 
    else:
        raise ValueError

    return props
