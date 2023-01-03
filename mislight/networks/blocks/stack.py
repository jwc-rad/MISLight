from copy import deepcopy
import numpy as np
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn

from monai.networks.layers.factories import Pool
from monai.networks.layers.utils import get_act_layer

from mislight.networks.layers import PadConvolution

class StackedConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        stride: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        padding_type: str = "zeros",
        pooling: str = None,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        conv0 = PadConvolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=1 if pooling else stride,
            kernel_size=kernel_size,
            padding_type=padding_type,
            act=act,
            norm=norm,
            dropout=dropout,
        )
        conv1 = PadConvolution(
            spatial_dims,
            out_channels,
            out_channels,
            strides=1,
            kernel_size=kernel_size,
            padding_type=padding_type,
            act=act,
            norm=norm,
            dropout=dropout,
        )

        m = [conv0]
        m += [conv1 for _ in range(num_convs - 1)]
        
        if pooling:
            m += [Pool[pooling, spatial_dims](stride)]
        
        self.convs = nn.Sequential(*m)

    def forward(self, x):
        return self.convs(x)
    
class StackedConvResidualBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        stride: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        padding_type: str = "zeros",
        pooling: str = None,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        assert num_convs > 1
        super().__init__()
        conv0 = PadConvolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=1 if pooling else stride,
            kernel_size=kernel_size,
            padding_type=padding_type,
            act=act,
            norm=norm,
            dropout=dropout,
        )
        conv1 = PadConvolution(
            spatial_dims,
            out_channels,
            out_channels,
            strides=1,
            kernel_size=kernel_size,
            padding_type=padding_type,
            act=act,
            norm=norm,
            dropout=dropout,
        )
        conv2 = PadConvolution(
            spatial_dims,
            out_channels,
            out_channels,
            strides=1,
            kernel_size=kernel_size,
            padding_type=padding_type,
            act=None,
            norm=norm,
            dropout=None,
        )
        
        m = [conv0]
        m += [conv1 for _ in range(num_convs - 2)]
        m += [conv2]
        
        if pooling:
            m += [Pool[pooling, spatial_dims](stride)]
        
        self.convs = nn.Sequential(*m)
        
        if (in_channels != out_channels) or not (all(i==1 for i in stride) if hasattr(stride, '__iter__') else stride==1):
            self.skip = PadConvolution(
                spatial_dims,
                in_channels,
                out_channels,
                strides=stride,
                kernel_size=1,
                padding_type=padding_type,
                act=None,
                norm=norm,
                dropout=None,
            )
        else:
            self.skip = lambda x: x
        
        self.nonlin = get_act_layer(act)
        
    def forward(self, x):
        residual = x
        
        out = self.convs(x)
        residual = self.skip(residual)
        
        out += residual

        return self.nonlin(out)
    
class StackedBlocks(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        stride: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        padding_type: str = "zeros",
        pooling: str = None,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        block: nn.Module = StackedConvBlock,
        num_blocks: int = 1,            
    ):
        super().__init__()

        if num_blocks > 0:
            self.convs = nn.Sequential(
                block(spatial_dims, in_channels, out_channels, num_convs, stride, kernel_size, padding_type, pooling, norm, act, dropout),
                *[block(spatial_dims, out_channels, out_channels, num_convs, 1, kernel_size, padding_type, pooling, norm, act, dropout) for _ in range(num_blocks - 1)]
            )
        else:
            self.convs = lambda x: x

    def forward(self, x):
        return self.convs(x)
    
class ResidualStackedBlocks(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        stride: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        padding_type: str = "zeros",
        pooling: str = None,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        block: nn.Module = StackedConvBlock,
        num_blocks: int = 1,            
    ):
        assert num_blocks > 0
        super().__init__()

        block0 = block(spatial_dims, in_channels, out_channels, num_convs, stride, kernel_size, padding_type, pooling, norm, act, dropout)
        block1 = block(spatial_dims, out_channels, out_channels, num_convs, 1, kernel_size, padding_type, pooling, norm, act, dropout)
        block2 = block(spatial_dims, out_channels, out_channels, num_convs, 1, kernel_size, padding_type, pooling, norm, None, None)

        m = [block0]
        if num_blocks > 1:
            m += [block1 for _ in range(num_blocks - 2)]
            m += [block2]
        self.convs = nn.Sequential(*m)

        if (in_channels != out_channels) or not (all(i==1 for i in stride) if hasattr(stride, '__iter__') else stride==1):
            self.skip = PadConvolution(
                spatial_dims,
                in_channels,
                out_channels,
                strides=stride,
                kernel_size=1,
                padding_type=padding_type,
                act=None,
                norm=norm,
                dropout=None,
            )
        else:
            self.skip = lambda x: x
        
        self.nonlin = get_act_layer(act)
            
    def forward(self, x):
        residual = x
        
        out = self.convs(x)
        residual = self.skip(residual)
        
        out += residual

        return self.nonlin(out)