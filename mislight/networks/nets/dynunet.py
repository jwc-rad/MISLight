from typing import List, Optional, Sequence, Tuple, Type, Union

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock #, UnetUpBlock
from monai.networks.layers.factories import Act, split_args
from mislight.networks.blocks.dynunet_block import UnetUpBlock

class DynUNetEncoder(nn.Module):
    '''
    one input block, `n` downsample blocks, one bottleneck and `n+1` upsample blocks
    '''
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
        num_blocks: Union[Sequence[int], int] = 1,
        res_block: Union[Sequence[bool], bool] = False,
        max_filters: int = 512,
        default_return_skips: bool = True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        self.default_return_skips = default_return_skips
        if isinstance(filters, int):
            self.filters = [min(filters * (2 ** i), max_filters) for i in range(len(strides))]
        else:
            self.filters = filters
        self.check_filters()
        if isinstance(num_blocks, int):
            self.num_blocks = [num_blocks]*(len(strides))
        else:
            self.num_blocks = num_blocks
        self.check_num_blocks()
        if isinstance(res_block, Sequence):
            self.conv_blocks = [UnetResBlock if x else UnetBasicBlock for x in res_block]
        else:
            self.conv_blocks = [UnetResBlock]*len(strides) if res_block else [UnetBasicBlock]*len(strides)
        self.check_conv_blocks()
        downsamples = []
        
        in_filters = [self.in_channels] + self.filters[:-1]
        out_filters = self.filters
        for i in range(len(self.strides)):
            ic = in_filters[i]
            oc = out_filters[i]
            k = self.kernel_size[i]
            s = self.strides[i]
            nb = self.num_blocks[i]
            cb = self.conv_blocks[i]
            
            downsamples.append(self.get_stacked_blocks(ic, oc, k, s, cb, nb))
            
        self.downsamples = nn.ModuleList(downsamples)
            
    def check_filters(self):
        filters = self.filters
        if len(filters) < len(self.strides):
            raise ValueError("length of filters should be no less than the length of strides.")
        else:
            self.filters = filters[: len(self.strides)]
    def check_num_blocks(self):
        num_blocks = self.num_blocks
        if len(num_blocks) < len(self.strides):
            raise ValueError("length of num_blocks should be no less than the length of strides.")
        else:
            self.num_blocks = num_blocks[: len(self.strides)]
    def check_conv_blocks(self):
        conv_blocks = self.conv_blocks
        if len(conv_blocks) < len(self.strides):
            raise ValueError("length of conv_blocks should be no less than the length of strides.")
        else:
            self.conv_blocks = conv_blocks[: len(self.strides)]
            
    def get_stacked_blocks(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        conv_block: Type[nn.Module],
        num_block: int = 1,
    ):
        if num_block == 0:
            layers = [nn.Identity()]
        else:
            layers = []
        for i in range(num_block):
            if i == 0:
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)    
            else:
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": out_channels,
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": 1,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.Sequential(*layers)
    
    def forward(self, x, layers=[], encode_only=False, return_skips=None):
        if return_skips is None:
            return_skips = self.default_return_skips
        
        if not encode_only:
            skips = []
            feat = x
            for m in self.downsamples:
                feat  = m(feat)
                skips.append(feat)
            if return_skips:
                output = skips
            else:
                output = skips[-1]
                
        if len(layers) > 0:
            feat = x
            feats = []
            cnt = 0
            if cnt in layers:
                feats.append(feat)
            for m in self.downsamples:
                for layer in m:
                    feat = layer(feat)
                    cnt += 1
                    if cnt in layers:
                        feats.append(feat)
            if encode_only:
                return feats
            else:
                return output, feats
        else:
            return output
        
class DynUNetDecoder(nn.Module):
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
        
        self.conv_block = UnetUpBlock

        upsamples = []
        
        in_filters = self.filters[:-1]
        out_filters = self.filters[1:] 
        for i in range(len(self.strides)):
            params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_filters[i],
                    "out_channels": out_filters[i],
                    "kernel_size": self.kernel_size[i],
                    "stride": self.strides[i],
                    "upsample_kernel_size": self.upsample_kernel_size[i],
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    "mode": self.mode,
                    "interp_mode": self.interp_mode,
                    "align_corners": self.align_corners,
                    "resize_first": self.resize_first,
                }
            upsamples.append(self.conv_block(**params))            
        self.upsamples = nn.ModuleList(upsamples)  
        
        self.head = UnetOutBlock(self.spatial_dims, self.filters[-1], self.out_channels, dropout=self.dropout)
        if head_act is None:
            self.head_act = nn.Identity()
        else:
            _act, _act_args = split_args(head_act)
            self.head_act = Act[_act](**_act_args)
        
    def forward(self, skips):
        skips = skips[::-1]
        
        x = skips[0]
        
        for skip, up in zip(skips[1:], self.upsamples):
            x = up(x, skip)
        
        x = self.head_act(self.head(x))
        
        return x
    
class DynUNet(nn.Module):
    """Adapted from monai.networks.nets.DynUNet
    Encoder - Decoder format, no deep_supervision
    """
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
            dropout, norm_name, act_name, num_blocks, res_block, max_filters,
        )
        self.decoder = DynUNetDecoder(
            spatial_dims, out_channels, kernel_size[1:][::-1], strides[1:][::-1], self.encoder.filters[::-1], None,
            dropout, norm_name, act_name, mode, interp_mode, align_corners, resize_first, head_act,
        )
        
    def forward(self, x, layers=[], encode_only=False):
        if len(layers) > 0:
            return self.encoder(x, layers, encode_only)
        else:
            skips = self.encoder(x)
            out = self.decoder(skips)
            return out