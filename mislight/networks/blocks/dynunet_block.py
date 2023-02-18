from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, get_conv_layer

from mislight.networks.blocks.convolutions import ResizeConv, SubpixelConv

class UnetUpBlock(nn.Module):
    """
    Modified from monai.networks.blocks.dynunet_block.UnetUpBlock
    Added upsampling method with interpolation
    Args:
        mode: {"deconv", "resizeconv"}.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[tuple, str],
        act_name: Union[tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[tuple, str, float]] = None,
        trans_bias: bool = False,
        padding_mode: str = "zeros",
        upsample_mode: str = "deconv",
        **upsample_kwargs,
    ):
        super().__init__()
        if upsample_mode == 'deconv':
            self.upsample = get_conv_layer(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                stride=stride,
                dropout=dropout,
                bias=trans_bias,
                act=None,
                norm=None,
                conv_only=False,
                is_transposed=True,
            )
        elif upsample_mode == 'resizeconv':
            self.upsample = ResizeConv(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                scale_factor=stride,
                dropout=dropout,
                bias=trans_bias,
                act=None,
                norm=None,
                conv_only=False,
                padding_mode=padding_mode,
                **upsample_kwargs
            )
        elif upsample_mode == 'subpixelconv':
            self.upsample = SubpixelConv(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=upsample_kernel_size,
                scale_factor=stride,
                dropout=dropout,
                bias=trans_bias,
                act=None,
                norm=None,
                conv_only=False,
                padding_mode=padding_mode,
                **upsample_kwargs
            )
        else:
            raise ValueError(f"upsample mode '{upsample_mode}' is not recognized.")
        
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.upsample(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out