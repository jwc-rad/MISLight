from copy import deepcopy
import numpy as np
import torch
from torch import nn

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
    
class ResizeConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, scale_factor, bias=True, padding_mode='zeros', mode='nearest', align_corners=True, dim=3):
        assert dim in [2,3]
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

        if dim == 3:
            conv = nn.Conv3d
        elif dim == 2:
            conv = nn.Conv2d
        self.conv = conv(input_channels, output_channels, kernel_size, 1, padding='same', padding_mode=padding_mode, bias=bias)
        
    def forward(self, x):
        sh = np.array(x.shape[2:])
        scale = np.array(self.scale_factor)
        newsize = tuple(sh * scale)
        y = nn.functional.interpolate(x, size=newsize, mode=self.mode, align_corners=self.align_corners)
        return self.conv(y)    
    
class ResizeConv3d(ResizeConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dim=3, **kwargs)
        
class ResizeConv2d(ResizeConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dim=2, **kwargs)

class DepthwiseSeparableConv(nn.Module):        
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, dim=3):
        assert dim in [2,3]
        super().__init__()
        if dim == 3:
            conv = nn.Conv3d
        elif dim == 2:
            conv = nn.Conv2d
        
        self.depthwise = conv(input_channels, input_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=input_channels, bias=bias)
        self.pointwise = conv(input_channels, output_channels, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableConv3d(DepthwiseSeparableConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dim=3, **kwargs)
        
class DepthwiseSeparableConv2d(DepthwiseSeparableConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dim=2, **kwargs)
