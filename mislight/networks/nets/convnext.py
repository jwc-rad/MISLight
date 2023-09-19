from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath

from monai.networks.layers.factories import Conv

from mislight.networks.blocks.convnext_block import ConvNeXtBlock


"""
Copied from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
Modified in MONAI format
"""

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(
        self,
        spatial_dims,
        in_channels = 3,
        num_classes=1000, 
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.,
        layer_scale_init_value=1e-6, 
        head_init_scale=1.,         
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        conv_type = Conv[Conv.CONV, self.spatial_dims]        
        einops_dims = ' '.join([x for i,x in enumerate(['h','w','d']) if i < spatial_dims])

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            conv_type(in_channels, dims[0], kernel_size=4, stride=4),
            Rearrange(f"b c {einops_dims} -> b {einops_dims} c"),
            nn.LayerNorm(dims[0], eps=1e-6),
            Rearrange(f"b {einops_dims} c -> b c {einops_dims}"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                Rearrange(f"b c {einops_dims} -> b {einops_dims} c"),
                nn.LayerNorm(dims[i], eps=1e-6),
                Rearrange(f"b {einops_dims} c -> b c {einops_dims}"),
                conv_type(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(spatial_dims=spatial_dims, dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        conv_type = Conv[Conv.CONV, self.spatial_dims]
        if isinstance(m, (conv_type, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x.flatten(start_dim=2).mean(2)) # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return x
    
    
convnext_params = {
    # model name: (depths, dims)
    "convnext_atto": ([2, 2, 6, 2], [40, 80, 160, 320]),
    "convnext_femto": ([2, 2, 6, 2], [48, 96, 192, 384]),
    "convnext_pico": ([2, 2, 6, 2], [64, 128, 256, 512]),
    "convnext_nano": ([2, 2, 8, 2], [80, 160, 320, 640]),
    "convnext_tiny": ([3, 3, 9, 3], [96, 192, 384, 768]),
    "convnext_small": ([3, 3, 27, 3], [96, 192, 384, 768]),
    "convnext_base": ([3, 3, 27, 3], [128, 256, 512, 1024]),
    "convnext_large": ([3, 3, 27, 3], [192, 384, 768, 1536]),
    "convnext_xlarge": ([3, 3, 27, 3], [256, 512, 1024, 2048]),
    "convnext_xxlarge": ([3, 4, 30, 3], [384, 768, 1536, 3072]),
}
    
class ConvNeXtBN(ConvNeXt):
    def __init__(
        self,
        model_name,
        spatial_dims=2,
        in_channels=3,
        num_classes=1000,
        **kwargs,
    ) -> None:
                
        # check if model_name is valid model
        if model_name not in convnext_params:
            model_name_string = ", ".join(convnext_params.keys())
            raise ValueError(f"invalid model_name {model_name} found, must be one of {model_name_string} ")

        # get depths, dims
        depths, dims = convnext_params[model_name]
        if not 'depths' in kwargs.keys():
            kwargs.update(depths=depths)
        if not 'dims' in kwargs.keys():
            kwargs.update(dims=dims)
            
        super().__init__(
            spatial_dims,
            in_channels,
            num_classes,
            **kwargs,
        )