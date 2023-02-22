from typing import List, Optional, Sequence, Tuple, Type, Union

import torch.nn as nn

from monai.networks.blocks.dynunet_block import get_output_padding, get_padding
from monai.networks.layers.factories import Act, split_args

from mislight.networks.blocks.convolutions import Convolution, ResizeConv, SubpixelConv
from mislight.networks.blocks.stack import StackedConvBasicBlock, StackedConvResidualBlock

"""
Reimplementation of ResnetGenerator from https://github.com/taesungp/contrastive-unpaired-translation
"""

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,  
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        filters: Union[Sequence[int], int] = 64,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        padding_mode: str = "zeros",
        num_blocks: int = 9,
        max_filters: int = 512,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.padding_mode = padding_mode
        self.num_blocks = num_blocks
        if isinstance(filters, Sequence):
            self.filters = filters
        else:
            self.filters = [min(filters * (2 ** i), max_filters) for i in range(len(strides))]
        self.check_with_strides('filters')

        model = []
        in_filters = [self.in_channels] + self.filters[:-1]
        out_filters = self.filters
        for i in range(len(self.strides)):
            params = {
                'spatial_dims': self.spatial_dims,
                'in_channels': in_filters[i],
                'out_channels': out_filters[i],
                'num_convs': 1,
                'stride': self.strides[i],
                'kernel_size': self.kernel_size[i],
                'padding_mode': self.padding_mode,
                'norm': self.norm_name,
                'act': self.act_name,
                'dropout': self.dropout,
            }
            model.append(StackedConvBasicBlock(**params))

        params = {
            'spatial_dims': self.spatial_dims,
            'in_channels': self.filters[-1],
            'out_channels': self.filters[-1],
            'num_convs': 2,
            'stride': 1,
            'kernel_size': self.kernel_size[-1],
            'padding_mode': self.padding_mode,
            'norm': self.norm_name,
            'act': self.act_name,
            'dropout': self.dropout,
        }
            
        for _ in range(self.num_blocks):
            model += [StackedConvResidualBlock(**params)]
            
        self.model = nn.Sequential(*model)
            
    def check_with_strides(self, attr):
        x = getattr(self, attr)
        if len(x) < len(self.strides):
            raise ValueError(f"length of {x} should be no less than the length of strides.")
        else:
            setattr(self, attr, x[:len(self.strides)])
    
    def forward(self, x, layers=[], encode_only=False):        
        if len(layers) > 0:
            feat = x
            feats = []
            cnt = 0
            if cnt in layers:
                feats.append(feat)
            for layer in self.model:
                feat = layer(feat)
                cnt += 1
                if cnt in layers:
                    feats.append(feat)
                if cnt == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            return self.model(x)

class ResNetDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int, 
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        filters: Sequence[int],
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        padding_mode: str = "zeros",
        upsample_mode: str = "deconv",
        head_act: Optional[Union[Tuple, str]] = None,
        **upsample_kwargs,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = filters
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.padding_mode = padding_mode
        self.upsample_mode = upsample_mode
        self.upsample_kwargs = upsample_kwargs
        
        if self.upsample_mode == 'deconv':
            self.get_conv = Convolution
        elif self.upsample_mode == 'resizeconv':
            self.get_conv = ResizeConv
        elif self.upsample_mode == 'subpixelconv':
            self.get_conv = SubpixelConv
        else:
            raise ValueError(f"upsample mode '{upsample_mode}' is not recognized.")

        model = []
        
        in_filters = self.filters
        out_filters = self.filters[1:] + [self.out_channels]
        for i in range(len(self.strides)):
            if i == len(self.strides) - 1 and self.strides[i] == 1:
                padding = get_padding(self.kernel_size[i], self.strides[i])              
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "strides": self.strides[i],
                    "kernel_size": self.kernel_size[i],
                    "act": None,
                    "norm": None,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "padding": padding,
                    "padding_mode": self.padding_mode,
                }
                model.append(Convolution(**params))
                continue
            
            if self.upsample_mode == 'deconv': 
                padding = get_padding(self.kernel_size[i], self.strides[i])
                output_padding = get_output_padding(self.kernel_size[i], self.strides[i], padding)               
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "strides": self.strides[i],
                    "kernel_size": self.kernel_size[i],
                    "act": self.act_name if i < len(self.strides) - 1 else None,
                    "norm": self.norm_name if i < len(self.strides) - 1 else None,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "is_transposed": True,
                    "padding": padding,
                    "output_padding": output_padding,
                    "padding_mode": 'zeros',
                }
                model.append(self.get_conv(**params))
            elif self.upsample_mode == 'resizeconv':
                padding = get_padding(self.kernel_size[i], 1)
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "scale_factor": self.strides[i],
                    "kernel_size": self.kernel_size[i],
                    "act": self.act_name if i < len(self.strides) - 1 else None,
                    "norm": self.norm_name if i < len(self.strides) - 1 else None,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "padding": padding,
                    "padding_mode": self.padding_mode,
                }
                params.update(self.upsample_kwargs)
                model.append(self.get_conv(**params)) 
            elif self.upsample_mode == 'subpixelconv':
                padding = get_padding(self.kernel_size[i], 1)
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "scale_factor": self.strides[i],
                    "kernel_size": self.kernel_size[i],
                    "act": self.act_name if i < len(self.strides) - 1 else None,
                    "norm": self.norm_name if i < len(self.strides) - 1 else None,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "padding": padding,
                    "padding_mode": self.padding_mode,
                }
                params.update(self.upsample_kwargs)
                model.append(self.get_conv(**params))            
          
        if head_act is not None:
            _act, _act_args = split_args(head_act)
            head_act = Act[_act](**_act_args)
            model.append(head_act)
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNetDecoderWithBottleneck(nn.Module):
    """Bottleneck blocks + ResNetDecoder"""
    def __init__(
        self,
        spatial_dims: int, 
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        filters: Sequence[int],
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        padding_mode: str = "zeros",
        upsample_mode: str = "deconv",
        head_act: Optional[Union[Tuple, str]] = None,
        num_blocks: int = 9,
        **upsample_kwargs,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = filters
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.padding_mode = padding_mode
        self.upsample_mode = upsample_mode
        self.upsample_kwargs = upsample_kwargs
        self.num_blocks = num_blocks
        
        if self.upsample_mode == 'deconv':
            self.get_conv = Convolution
        elif self.upsample_mode == 'resizeconv':
            self.get_conv = ResizeConv
        elif self.upsample_mode == 'subpixelconv':
            self.get_conv = SubpixelConv
        else:
            raise ValueError(f"upsample mode '{upsample_mode}' is not recognized.")

        model = []
        params = {
            'spatial_dims': self.spatial_dims,
            'in_channels': self.filters[0],
            'out_channels': self.filters[0],
            'num_convs': 2,
            'stride': 1,
            'kernel_size': self.kernel_size[0],
            'padding_mode': self.padding_mode,
            'norm': self.norm_name,
            'act': self.act_name,
            'dropout': self.dropout,
        }
            
        for _ in range(self.num_blocks):
            model += [StackedConvResidualBlock(**params)]
                    
        in_filters = self.filters
        out_filters = self.filters[1:] + [self.out_channels]
        for i in range(len(self.strides)):
            if i == len(self.strides) - 1 and self.strides[i] == 1:
                padding = get_padding(self.kernel_size[i], self.strides[i])              
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "strides": self.strides[i],
                    "kernel_size": self.kernel_size[i],
                    "act": None,
                    "norm": None,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "padding": padding,
                    "padding_mode": self.padding_mode,
                }
                model.append(Convolution(**params))
                continue
            
            if self.upsample_mode == 'deconv': 
                padding = get_padding(self.kernel_size[i], self.strides[i])
                output_padding = get_output_padding(self.kernel_size[i], self.strides[i], padding)               
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "strides": self.strides[i],
                    "kernel_size": self.kernel_size[i],
                    "act": self.act_name if i < len(self.strides) - 1 else None,
                    "norm": self.norm_name if i < len(self.strides) - 1 else None,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "is_transposed": True,
                    "padding": padding,
                    "output_padding": output_padding,
                    "padding_mode": 'zeros',
                }
                model.append(self.get_conv(**params))
            elif self.upsample_mode == 'resizeconv':
                padding = get_padding(self.kernel_size[i], 1)
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "scale_factor": self.strides[i],
                    "kernel_size": self.kernel_size[i],
                    "act": self.act_name if i < len(self.strides) - 1 else None,
                    "norm": self.norm_name if i < len(self.strides) - 1 else None,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "padding": padding,
                    "padding_mode": self.padding_mode,
                }
                params.update(self.upsample_kwargs)
                model.append(self.get_conv(**params)) 
            elif self.upsample_mode == 'subpixelconv':
                padding = get_padding(self.kernel_size[i], 1)
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "scale_factor": self.strides[i],
                    "kernel_size": self.kernel_size[i],
                    "act": self.act_name if i < len(self.strides) - 1 else None,
                    "norm": self.norm_name if i < len(self.strides) - 1 else None,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "padding": padding,
                    "padding_mode": self.padding_mode,
                }
                params.update(self.upsample_kwargs)
                model.append(self.get_conv(**params))            
          
        if head_act is not None:
            _act, _act_args = split_args(head_act)
            head_act = Act[_act](**_act_args)
            model.append(head_act)
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class ResNetGenerator(nn.Module):
    def __init__(
        self,
        spatial_dims: int, 
        in_channels: int,  
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        filters: Union[Sequence[int], int] = 64,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        padding_mode: str = "zeros",
        num_blocks: int = 9,
        max_filters: int = 512,
        upsample_mode: str = "deconv",  
        head_act: Optional[Union[Tuple, str]] = None,
        **upsample_kwargs,
    ):    
        super().__init__()
        self.encoder = ResNetEncoder(
            spatial_dims, in_channels, kernel_size, strides, filters,
            dropout, norm_name, act_name, padding_mode, num_blocks, max_filters,
        )
        self.decoder = ResNetDecoder(
            spatial_dims, out_channels, kernel_size[::-1], strides[::-1], self.encoder.filters[::-1],
            dropout, norm_name, act_name, padding_mode, upsample_mode, head_act, **upsample_kwargs,
        )
        
    def forward(self, x, layers=[], encode_only=False):
        if len(layers) > 0:
            return self.encoder(x, layers, encode_only)
        else:
            bottleneck = self.encoder(x)
            out = self.decoder(bottleneck)
            return out