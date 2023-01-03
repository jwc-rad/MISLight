from copy import deepcopy
import functools
import math
import numpy as np
from typing import Optional, Sequence, Tuple, Union
import torch
from torch import nn

from monai.networks.blocks import ADN
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv, Pad

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
    
class PadConvolution(nn.Sequential):
    '''Added padding in front of monai.networks.blocks.convolutions.Convolution
    '''
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        padding_type: str = "zeros",
        output_padding: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.dimensions]
            
        if padding_type == 'reflect':
            if self.dimensions == 1:
                pad_type = nn.ReflectionPad1d
            elif self.dimensions == 2:
                pad_type = nn.ReflectionPad2d
            else:
                raise NotImplementedError(f'padding {padding_type} is not implemented for dims {self.dimensions}')
        elif padding_type == 'replicate':
            pad_type = Pad[Pad.REPLICATIONPAD, self.dimensions]
        else:
            pad_type = Pad[Pad.CONSTANTPAD, self.dimensions]
            pad_type = functools.partial(pad_type, value=0)        
        
        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
            )
            self.add_module("conv", conv)
        else:
            pads = pad_type(padding)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.add_module("pad", pads)
            self.add_module("conv", conv)

        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.dimensions,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )
    
class PadConvolution3d(PadConvolution):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
        
class PadConvolution2d(PadConvolution):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)
        
    
class ResizeConv(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size, scale_factor, bias=True, padding_mode='zeros', mode='nearest', align_corners=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

        conv = Conv[Conv.CONV, spatial_dims]
        self.conv = conv(in_channels, out_channels, kernel_size, 1, padding='same', padding_mode=padding_mode, bias=bias)
        
    def forward(self, x):
        sh = np.array(x.shape[2:])
        scale = np.array(self.scale_factor)
        newsize = tuple(sh * scale)
        y = nn.functional.interpolate(x, size=newsize, mode=self.mode, align_corners=self.align_corners)
        return self.conv(y)    
    
class ResizeConv3d(ResizeConv):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)
        
class ResizeConv2d(ResizeConv):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)
        
        
class PixelShuffleUpsampler(nn.Sequential):
    '''added PixelShuffle between C-NDA of monai.networks.blocks.convolutions.Convolution
    '''
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        scale: int = 2,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        bias: bool = True,
        padding: Optional[Union[Sequence[int], int]] = None,
        padding_type: str = "zeros",
    ) -> None:
        super().__init__()
        
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.scale = scale
        
        if padding is None:
            padding = same_padding(kernel_size, 1)
        conv_type = Conv[Conv.CONV, self.dimensions]
            
        if padding_type == 'reflect':
            if self.dimensions == 1:
                pad_type = nn.ReflectionPad1d
            elif self.dimensions == 2:
                pad_type = nn.ReflectionPad2d
            else:
                raise NotImplementedError(f'padding {padding_type} is not implemented for dims {self.dimensions}')
        elif padding_type == 'replicate':
            pad_type = Pad[Pad.REPLICATIONPAD, self.dimensions]
        else:
            pad_type = Pad[Pad.CONSTANTPAD, self.dimensions]
            pad_type = functools.partial(pad_type, value=0)        
        
        pads = pad_type(padding)
        conv = conv_type(
            in_channels,
            in_channels * scale * scale,
            kernel_size=kernel_size,
            padding=0,
            bias=bias,
        )
        self.add_module("pad", pads)
        self.add_module("conv", conv)
        self.add_module("pixelshuffle", nn.PixelShuffle(scale))

        if act is None and norm is None and dropout is None:
            return
        else:
            self.add_module(
                "adn",
                ADN(
                    ordering=adn_ordering,
                    in_channels=in_channels,
                    act=act,
                    norm=norm,
                    norm_dim=self.dimensions,
                    dropout=dropout,
                    dropout_dim=dropout_dim,
                ),
            )