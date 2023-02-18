from copy import deepcopy
import functools
import math
import numpy as np
from typing import Optional, Sequence, Tuple, Union
import torch
from torch import nn

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv, Pad, Pool
from monai.utils import ensure_tuple_rep

class Convolution(nn.Sequential):
    """
    extension of monai.networks.blocks.convolutions.Convolution
        - padding_mode
    """
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
        output_padding: Optional[Union[Sequence[int], int]] = None,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.spatial_dims]

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
                padding_mode=padding_mode,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )

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
                norm_dim=self.spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )
        
            
class ResizeConv(nn.Sequential):
    """ResizeConv + ADN
    Ref: https://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        scale_factor: Union[Sequence[int], int] = 2,
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
        padding: Optional[Union[Sequence[int], int]] = None,
        padding_mode: str = "zeros",
        interp_mode: str = "nearest", 
        align_corners: Optional[bool] = None, 
        resize_first = True,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        if padding is None:
            padding = same_padding(kernel_size, dilation)
            
        up = nn.Upsample(scale_factor=scale_factor, mode=interp_mode, align_corners=align_corners)
            
        conv_type = Conv[Conv.CONV, self.spatial_dims]
        conv = conv_type(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        
        if resize_first:
            self.add_module("resize", up)
            self.add_module("conv", conv)
        else:
            self.add_module("conv", conv)
            self.add_module("resize", up)

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
                norm_dim=self.spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )
        
class SubpixelConv(nn.Sequential):
    """SubpixelConv (conv + pixelshuffle) + ADN
    Adapted from monai.networks.blocks.SubpixelUpsample.
    Ref: Shi et al., 2016, "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network."
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        scale_factor: Union[Sequence[int], int] = 2,
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
        padding: Optional[Union[Sequence[int], int]] = None,
        padding_mode: str = "zeros",
        apply_pad_pool: bool = True,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor[0] if isinstance(scale_factor, Sequence) else scale_factor
        if padding is None:
            padding = same_padding(kernel_size, dilation)
            
        conv_type = Conv[Conv.CONV, self.spatial_dims]
        conv = conv_type(
            in_channels,
            out_channels * (self.scale_factor**self.spatial_dims),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.add_module("conv", conv)
        
        pixelshuffle = nn.PixelShuffle(self.scale_factor)
        self.add_module("pixelshuffle", pixelshuffle)

        if apply_pad_pool:
            pool_type = Pool[Pool.AVG, self.spatial_dims]
            pad_type = Pad[Pad.CONSTANTPAD, self.spatial_dims]

            pad_ = pad_type(padding=(self.scale_factor - 1, 0) * self.spatial_dims, value=0.0)
            pool_ = pool_type(kernel_size=self.scale_factor, stride=1)
            self.add_module("pad", pad_)
            self.add_module("pool", pool_)

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
                norm_dim=self.spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )