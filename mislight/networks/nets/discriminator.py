from copy import deepcopy
import functools
import numpy as np
from typing import Optional, Sequence, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution

from mislight.utils.misc import safe_repeat

class PatchGANDiscriminator(nn.Module):
    '''Re-implementation of NLayerDiscriminator from https://github.com/taesungp/contrastive-unpaired-translation
    
    n_layers & receptive field: 3 = 70, 4 = 142, 5 = 286
    
    '''
    def __init__(
        self,
        spatial_dims: int,
        channels: Union[Sequence[int], int],
        base_num_features: int,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.2}),
        n_layers: int = 3,
    ):
        super().__init__()

        in_channels = safe_repeat(channels, 1)[0]
        kw = [4,] * (n_layers + 2)
        padw = [1,] * (n_layers + 2)
        strw = [2,] * (n_layers) + [1,] * 2
        
        sequence = [Convolution(spatial_dims, in_channels, base_num_features, kernel_size=kw[0], strides=strw[0], padding=padw[0], norm=None, act=act)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [Convolution(spatial_dims, base_num_features * nf_mult_prev, base_num_features * nf_mult, kernel_size=kw[n], strides=strw[n], padding=padw[n], norm=norm, act=act)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [Convolution(spatial_dims, base_num_features * nf_mult_prev, base_num_features * nf_mult, kernel_size=kw[-2], strides=strw[-2], padding=padw[-2], norm=norm, act=act)]

        sequence += [Convolution(spatial_dims, base_num_features * nf_mult, 1, kernel_size=kw[-1], strides=strw[-1], padding=padw[-1], conv_only = True)]
        
        self.model = nn.Sequential(*sequence)
        self.receptive_field = calculate_receptive_field(1, kw, strw)
                
    def forward(self, x):
        return self.model(x)
    
def calculate_receptive_field(last_size, kernels, strides):
    assert len(kernels) == len(strides)
    field = (last_size - 1) * strides[-1] + kernels[-1]
    if len(strides) == 1:
        return field
    else:
        return calculate_receptive_field(field, kernels[:-1], strides[:-1])