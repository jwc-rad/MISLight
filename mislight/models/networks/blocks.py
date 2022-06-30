from copy import deepcopy
import numpy as np
import torch
from torch import nn

from .layers import Identity

'''From https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/custom_modules/conv_blocks.py
'''
class CNDA(nn.Module):
    ''' (Conv) -- (Normal) -- (Dropout) -- (Activation)
    '''
    def __init__(self, input_channels, output_channels, kernel_size, props):
        super().__init__()
        props = deepcopy(props)  # props is a dict and mutable, so we deepcopy to be safe.

        self.conv = props['conv_op'](input_channels, output_channels, kernel_size,
                                     padding=[(i - 1) // 2 for i in kernel_size],
                                     **props['conv_op_kwargs'])

        if props['norm_op'] is not None:
            self.norm = props['norm_op'](output_channels, **props['norm_op_kwargs'])
        else:
            self.norm = Identity()
            
        if props['dropout_op'] is not None:
            self.do = props['dropout_op'](**props['dropout_op_kwargs'])
        else:
            self.do = Identity()
            
        if props['nonlin'] is not None:
            self.nonlin = props['nonlin'](**props['nonlin_kwargs'])
        else:
            self.nonlin = Identity()
        
        self.all = nn.Sequential(self.conv, self.norm, self.do, self.nonlin)

    def forward(self, x):
        return self.all(x)
        
class StackedConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, props, num_convs, stride=None):
        super().__init__()
        props = deepcopy(props)  # props is a dict and mutable, so we deepcopy to be safe.
        props_first = deepcopy(props)

        if stride is not None:
            props_first['conv_op_kwargs']['stride'] = stride

        self.convs = nn.Sequential(
            CNDA(input_channels, output_channels, kernel_size, props_first),
            *[CNDA(output_channels, output_channels, kernel_size, props) for _ in range(num_convs - 1)]
        )

    def forward(self, x):
        return self.convs(x)
        
class StackedConvResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, props, num_convs, stride=None):
        assert num_convs > 1
        super().__init__()
        props = deepcopy(props)  # props is a dict and mutable, so we deepcopy to be safe.
        props_first = deepcopy(props)
        props_last = deepcopy(props)
        
        if stride is not None:
            props_first['conv_op_kwargs']['stride'] = stride
        props_last['dropout_op'] = None
        props_last['nonlin'] = None
            
        self.convs = nn.Sequential(
            CNDA(input_channels, output_channels, kernel_size, props_first),
            *[CNDA(output_channels, output_channels, kernel_size, props) for _ in range(num_convs - 2)],
            CNDA(output_channels, output_channels, kernel_size, props_last)
        )

        if (stride is not None and any((i != 1 for i in stride))) or (input_channels != output_channels):
            stride_here = stride if stride is not None else 1
            self.skip = nn.Sequential(props['conv_op_base'](input_channels, output_channels, 1, stride_here, bias=False),
                                      props['norm_op'](output_channels, **props['norm_op_kwargs']))
        else:
            self.skip = lambda x: x
            
        self.nonlin = props['nonlin'](**props['nonlin_kwargs'])
        
    def forward(self, x):
        residual = x
        
        out = self.convs(x)
        residual = self.skip(residual)
        
        out += residual

        return self.nonlin(out)

class StackedBlocks(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, props, num_blocks, num_convs, block=StackedConvBlock, first_stride=None):
        super().__init__()
        props = deepcopy(props)  # network_props is a dict and mutable, so we deepcopy to be safe.

        self.convs = nn.Sequential(
            block(input_channels, output_channels, kernel_size, props, num_convs, first_stride),
            *[block(output_channels, output_channels, kernel_size, props, num_convs) for _ in range(num_blocks - 1)]
        )

    def forward(self, x):
        return self.convs(x)
