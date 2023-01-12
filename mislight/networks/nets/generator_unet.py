from copy import deepcopy
import functools
import math
import numpy as np
from typing import Optional, Sequence, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

from monai.networks.nets import DynUNet

from mislight.networks.utils import init_weights

class DynUNetGenerator(DynUNet):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False,
        tail_act: bool = False,
        **initialize_weights_kwargs
    ):
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            strides,
            upsample_kernel_size,
            filters,
            dropout,
            norm_name,
            act_name,
            deep_supervision,
            deep_supr_num,
            res_block,
            trans_bias,            
        )
        
        if tail_act:
            self.tail_act = nn.Tanh()
        
        self.initialize_weights(**initialize_weights_kwargs)
        
    def initialize_weights(self, init_type='xavier', **kwargs):
        init_func = functools.partial(init_weights, init_type=init_type, **kwargs)
        self.apply(init_func)  
        
    def forward(self, x, layers=[], encode_only=False):
        if not encode_only:
            out = self.skip_layers(x)
            out = self.output_block(out)
            if hasattr(self, 'tail_act'):
                out = self.tail_act(out)            
        
        if len(layers) > 0:
            feat = x
            feats = []
            cnt = 0
            if cnt in layers:
                feats.append(feat)
            m = self.skip_layers
            while hasattr(m, 'next_layer'):
                if hasattr(m, 'downsample'):
                    feat = m.downsample(feat)
                else:
                    feat = m(feat)
                cnt += 1
                if cnt in layers:
                    feats.append(feat)
                m = m.next_layer
            
            if encode_only:
                return feats
            else:               
                return out, feats
        else:
            return out