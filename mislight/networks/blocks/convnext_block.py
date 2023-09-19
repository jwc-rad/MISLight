from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath

from monai.networks.layers.factories import Conv

"""
Copied from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
Modified in MONAI format
"""

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(
        self,
        spatial_dims,
        dim, 
        drop_path=0., 
        layer_scale_init_value=1e-6,
    ) -> None:
        super().__init__()
        
        conv_type = Conv[Conv.CONV, spatial_dims]
        einops_dims = ' '.join([x for i,x in enumerate(['h','w','d']) if i < spatial_dims])
        
        self.dwconv = conv_type(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.rearrange1 = Rearrange(f"b c {einops_dims} -> b {einops_dims} c")
        self.rearrange2 = Rearrange(f"b {einops_dims} c -> b c {einops_dims}")
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)       
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.rearrange1(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = self.rearrange2(x)

        x = input + self.drop_path(x)
        return x