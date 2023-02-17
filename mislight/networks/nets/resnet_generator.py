from typing import List, Optional, Sequence, Tuple, Type, Union

import torch.nn as nn

from monai.networks.blocks.dynunet_block import get_output_padding, get_padding
from monai.networks.layers.factories import Act, split_args

from mislight.networks.blocks.convolutions import Convolution, ResizeConv
from .dynunet import DynUNetEncoder

class ResNetDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int, 
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        filters: Sequence[int],
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        mode: str = "deconv",
        interp_mode: str = "area",
        align_corners: Optional[bool] = None,
        resize_first: bool = True,
        head_act: Optional[Union[Tuple, str]] = None,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = filters
        if upsample_kernel_size is None:
            self.upsample_kernel_size = kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.mode = mode
        self.interp_mode = interp_mode
        self.align_corners = align_corners
        self.resize_first = resize_first
        
        if mode == 'deconv':
            self.get_conv = Convolution
        elif mode == 'resizeconv':
            self.get_conv = ResizeConv
        else:
            raise ValueError(f"upsample mode '{mode}' is not recognized.")

        upsamples = []
        
        in_filters = self.filters[:-1]
        out_filters = self.filters[1:] 
        for i in range(len(self.strides)):
            if mode == 'deconv': 
                padding = get_padding(self.strides[i], self.strides[i])
                output_padding = get_output_padding(self.strides[i], self.strides[i], padding)               
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "strides": self.strides[i],
                    "kernel_size": self.strides[i],
                    "act": self.act_name,
                    "norm": self.norm_name,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "is_transposed": True,
                    "padding": padding,
                    "output_padding": output_padding,
                }
                upsamples.append(self.get_conv(**params))
            elif mode == 'resizeconv':
                padding = get_padding(self.upsample_kernel_size[i], 1)
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "scale_factor": self.strides[i],
                    "kernel_size": self.upsample_kernel_size[i],
                    "act": self.act_name,
                    "norm": self.norm_name,
                    "dropout": self.dropout,
                    "conv_only": False,
                    "padding": padding,
                    "interp_mode": self.interp_mode,
                    "align_corners": self.align_corners,
                    "resize_first": self.resize_first,
                }
                upsamples.append(self.get_conv(**params))            
        self.upsamples = nn.Sequential(*upsamples)  
        
        self.head = Convolution(
            self.spatial_dims, 
            self.filters[-1], 
            self.out_channels, 
            kernel_size=1,
            strides=1,
            dropout=self.dropout,
            bias=True,
            act=None,
            norm=None,
            conv_only=False,
        )
        if head_act is None:
            self.head_act = nn.Identity()
        else:
            _act, _act_args = split_args(head_act)
            self.head_act = Act[_act](**_act_args)
        
    def forward(self, x):
        x = self.upsamples(x)        
        x = self.head_act(self.head(x))
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
        mode: str = "deconv",
        interp_mode: str = "area",
        align_corners: Optional[bool] = None,
        resize_first: bool = True,
        num_blocks: Union[Sequence[int], int] = 1,
        res_block: bool = False,
        max_filters: int = 512,
        head_act: Optional[Union[Tuple, str]] = None,
    ):    
        super().__init__()
        self.encoder = DynUNetEncoder(
            spatial_dims, in_channels, kernel_size, strides, filters, 
            dropout, norm_name, act_name, num_blocks, res_block, max_filters, False,
        )
        self.decoder = ResNetDecoder(
            spatial_dims, out_channels, kernel_size[1:][::-1], strides[1:][::-1], self.encoder.filters[::-1], None,
            dropout, norm_name, act_name, mode, interp_mode, align_corners, resize_first, head_act,
        )
        
    def forward(self, x, layers=[], encode_only=False):
        if len(layers) > 0:
            return self.encoder(x, layers, encode_only)
        else:
            bottleneck = self.encoder(x)
            out = self.decoder(bottleneck)
            return out