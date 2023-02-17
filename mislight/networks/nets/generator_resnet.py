from copy import deepcopy
import functools
import math
import numpy as np
from typing import Optional, Sequence, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F

from mislight.networks.layers import Identity, PadConvolution
from mislight.networks.blocks import StackedBlocks, StackedConvBlock, StackedConvResidualBlock
from mislight.networks.utils import InitializeWeights
from mislight.utils.misc import safe_repeat

class GenericSequentialEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        base_num_features: int,
        pools: Union[Sequence[int], Sequence[Sequence[int]]],
        kernel_sizes: Union[Sequence[int], Sequence[Sequence[int]]],
        padding_type: str = "zeros",
        pooling: str = None,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        feat_map_mul_on_downscale: Union[Sequence[int], int] = 2,
        block: Union[Sequence[nn.Module], nn.Module] = StackedConvBlock,
        block_bottleneck: nn.Module = StackedConvResidualBlock,
        num_blocks_per_stage: Union[Sequence[int], int] = 1,
        num_blocks_bottleneck: int = 1,
        num_convs_per_block: Union[Sequence[int], int] = 1,
        num_convs_per_block_bottleneck: int = 2,
        max_num_features: int = 512,
        default_return_skips: bool = True,
    ):    
        super().__init__()

        self.spatial_dims = spatial_dims
        self.default_return_skips = default_return_skips

        self.input = [Identity()]
        self.stages = []
        self.stage_out_features = []
        self.stage_pools = []
        self.stage_kernel_sizes = []
        self.stage_padding_types = []

        assert len(pools) == len(kernel_sizes)

        num_stages = len(kernel_sizes)
        
        if not isinstance(num_blocks_per_stage, (list, tuple)):
            num_blocks_per_stage = [num_blocks_per_stage] * num_stages
        else:
            assert len(num_blocks_per_stage) == num_stages
        if not isinstance(block, (list, tuple)):
            block = [block] * num_stages
        else:
            assert len(block) == num_stages
        if not isinstance(num_convs_per_block, (list, tuple)):
            num_convs_per_block = [num_convs_per_block] * num_stages
        else:
            assert len(num_convs_per_block) == num_stages
        if not isinstance(feat_map_mul_on_downscale, (list, tuple)):
            feat_map_mul_on_downscale = [feat_map_mul_on_downscale] * num_stages
        else:
            assert len(feat_map_mul_on_downscale) == num_stages
            
        cin_features = in_channels
            
        for stage in range(num_stages):
            cout_features = min(int(base_num_features * feat_map_mul_on_downscale[stage] ** stage), max_num_features)
            ckernel = kernel_sizes[stage]
            cpool = pools[stage]
            cblock = block[stage]
            cnum_convs = num_convs_per_block[stage]
            cnum_blocks = num_blocks_per_stage[stage]

            if stage == 0:
                cpadding_type = padding_type
            else:
                cpadding_type = 'zeros'
            
            cstage = StackedBlocks(spatial_dims, cin_features, cout_features, cnum_convs, cpool, ckernel, cpadding_type, pooling, norm, act, dropout, cblock, cnum_blocks)

            self.stages.append(cstage)
            self.stage_out_features.append(cout_features)
            self.stage_kernel_sizes.append(ckernel)
            self.stage_pools.append(cpool)
            self.stage_padding_types.append(cpadding_type)

            # update current_input_features
            cin_features = cout_features
            
            if stage == num_stages - 1:
                self.bottleneck = [block_bottleneck(spatial_dims, cin_features, cin_features, num_convs_per_block_bottleneck, 1, ckernel, padding_type, None, norm, act, dropout),] * num_blocks_bottleneck
            

        self.model_list = self.input + self.stages + self.bottleneck
        self.model = nn.Sequential(*self.model_list)
        self.out_features = cout_features

    def forward(self, x):
        """
        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        return self.model(x)

class GenericSequentialDecoder(nn.Module):
    def __init__(
        self,
        previous: nn.Module,
        out_channels: int,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        head_act: bool = False,
    ):    
        super().__init__()
        
        spatial_dims = previous.spatial_dims
        previous_stages = previous.stages        
        previous_stage_out_features = previous.stage_out_features
        previous_stage_pools = previous.stage_pools
        previous_stage_kernel_sizes = previous.stage_kernel_sizes
        previous_stage_padding_types = previous.stage_padding_types

        num_stages = len(previous_stages) - 1

        self.stage_out_features = previous_stage_out_features
        self.stage_pools = previous_stage_pools
        self.stage_kernel_sizes = previous_stage_kernel_sizes
        
        self.tus = []

        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_out_features[s + 1]
            features_skip = previous_stage_out_features[s]
            ckernel = previous_stage_kernel_sizes[s + 1]
            cpadding_type = previous_stage_padding_types[s + 1]
            cpool = previous_stage_pools[s + 1]

            self.tus.append(PadConvolution(spatial_dims, features_below, features_skip, cpool, ckernel, is_transposed=True, padding_type=cpadding_type, norm=norm, act=act, dropout=dropout))  
            
        ckernel = previous_stage_kernel_sizes[0]
        cpadding_type = previous_stage_padding_types[0]
        self.head = [PadConvolution(spatial_dims, features_skip, out_channels, 1, ckernel, conv_only=True, padding_type=cpadding_type)]
        if head_act:
            self.head += [nn.Tanh()]
        
        self.model_list = self.tus + self.head
        self.model = nn.Sequential(*self.model_list)
        
    def forward(self, x):       
        return self.model(x)

    
class ResnetGenerator(nn.Module, InitializeWeights):
    '''Re-implementation of ResnetGenerator from https://github.com/taesungp/contrastive-unpaired-translation
    '''
    def __init__(
        self,
        spatial_dims: int,
        channels: Union[Sequence[int], int],
        base_num_features: int,
        pools: Union[Sequence[int], Sequence[Sequence[int]]],
        kernel_sizes: Union[Sequence[int], Sequence[Sequence[int]]],
        padding_type: str = "zeros",
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        feat_map_mul_on_downscale: Union[Sequence[int], int] = 2,
        block_encoder: Union[Sequence[nn.Module], nn.Module] = StackedConvBlock,
        block_bottleneck: nn.Module = StackedConvResidualBlock,
        num_blocks_per_stage_encoder: Union[Sequence[int], int] = 1,
        num_blocks_bottleneck: int = 1,
        num_convs_per_block_encoder: Union[Sequence[int], int] = 1,
        num_convs_per_block_bottleneck: int = 2,
        max_num_features: int = 512,
        head_act: bool = False,
        **initialize_weights_kwargs,
    ):                
        super().__init__()

        channels = safe_repeat(channels, 2)
        in_channels = channels[0]
        out_channels = channels[1]
        
        encoder = GenericSequentialEncoder(spatial_dims, in_channels, base_num_features, pools, kernel_sizes, padding_type, None, norm, act, dropout, feat_map_mul_on_downscale, block_encoder, block_bottleneck, num_blocks_per_stage_encoder, num_blocks_bottleneck, num_convs_per_block_encoder, num_convs_per_block_bottleneck, max_num_features, False)
        decoder = GenericSequentialDecoder(encoder, out_channels, norm, act, dropout, head_act)

        self.model_list = encoder.model_list + decoder.model_list
        self.model = nn.Sequential(*self.model_list)
        
        self.initialize_weights(**initialize_weights_kwargs)
        
    def forward(self, x, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            return self.model(x)
        
        
# support upscaling
class GenericSequentialDecoderV2(nn.Module):
    def __init__(
        self,
        previous: nn.Module,
        out_channels: int,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        scale_factor: int = 4,
    ):    
        super().__init__()
        
        spatial_dims = previous.spatial_dims
        previous_stages = previous.stages        
        previous_stage_out_features = previous.stage_out_features
        previous_stage_pools = previous.stage_pools
        previous_stage_kernel_sizes = previous.stage_kernel_sizes
        previous_stage_padding_types = previous.stage_padding_types

        num_stages = len(previous_stages) - 1

        self.stage_out_features = previous_stage_out_features
        self.stage_pools = previous_stage_pools
        self.stage_kernel_sizes = previous_stage_kernel_sizes
        self.scale_factor = scale_factor
        
        self.tus = []

        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_out_features[s + 1]
            features_skip = previous_stage_out_features[s]
            ckernel = previous_stage_kernel_sizes[s + 1]
            cpadding_type = previous_stage_padding_types[s + 1]
            cpool = previous_stage_pools[s + 1]

            self.tus.append(PadConvolution(spatial_dims, features_below, features_skip, cpool, ckernel, is_transposed=True, padding_type=cpadding_type, norm=norm, act=act, dropout=dropout))  
            
        # upscaling
        num_ups = math.log(scale_factor, cpool)
        assert num_ups.is_integer()
        for _ in range(int(num_ups)):
            self.tus.append(PadConvolution(spatial_dims, features_skip, features_skip, cpool, ckernel, is_transposed=True, padding_type=cpadding_type, norm=norm, act=act, dropout=dropout))
            
        ckernel = previous_stage_kernel_sizes[0]
        cpadding_type = previous_stage_padding_types[0]
        self.tail_output = [PadConvolution(spatial_dims, features_skip, out_channels, 1, ckernel, conv_only=True, padding_type=cpadding_type)]
        
        self.model_list = self.tus + self.tail_output
        self.model = nn.Sequential(*self.model_list)
        
    def forward(self, x):       
        return self.model(x)

class ResnetGeneratorV2(nn.Module):
    '''Re-implementation of ResnetGenerator from https://github.com/taesungp/contrastive-unpaired-translation
    Changes:
        supports upscaling
    '''
    def __init__(
        self,
        spatial_dims: int,
        channels: Union[Sequence[int], int],
        base_num_features: int,
        pools: Union[Sequence[int], Sequence[Sequence[int]]],
        kernel_sizes: Union[Sequence[int], Sequence[Sequence[int]]],
        padding_type: str = "zeros",
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        feat_map_mul_on_downscale: Union[Sequence[int], int] = 2,
        block_encoder: Union[Sequence[nn.Module], nn.Module] = StackedConvBlock,
        block_bottleneck: nn.Module = StackedConvResidualBlock,
        num_blocks_per_stage_encoder: Union[Sequence[int], int] = 1,
        num_blocks_bottleneck: int = 1,
        num_convs_per_block_encoder: Union[Sequence[int], int] = 1,
        num_convs_per_block_bottleneck: int = 2,
        max_num_features: int = 512,
        scale_factor: int = 4,
    ):                
        super().__init__()

        channels = safe_repeat(channels, 2)
        in_channels = channels[0]
        out_channels = channels[1]
        
        encoder = GenericSequentialEncoder(spatial_dims, in_channels, base_num_features, pools, kernel_sizes, padding_type, None, norm, act, dropout, feat_map_mul_on_downscale, block_encoder, block_bottleneck, num_blocks_per_stage_encoder, num_blocks_bottleneck, num_convs_per_block_encoder, num_convs_per_block_bottleneck, max_num_features, False)
        decoder = GenericSequentialDecoderV2(encoder, out_channels, norm, act, dropout, scale_factor)

        self.model_list = encoder.model_list + decoder.model_list
        self.model = nn.Sequential(*self.model_list)
        
    def forward(self, x, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            return self.model(x)