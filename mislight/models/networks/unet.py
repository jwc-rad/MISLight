'''Adpated from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_modular_UNet.py                         
'''

from copy import deepcopy
import numpy as np
import torch
from torch import nn

from .layers import Identity, ResizeConv3d, ResizeConv2d
from .blocks import CNDA, StackedBlocks, StackedConvBlock, StackedConvResidualBlock

class GenericUNetEncoder(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props,
                 initial_conv=False, max_num_features=480, block=StackedConvBlock, num_convs_per_block=2, default_return_skips=True,):
        super().__init__()

        self.default_return_skips = default_return_skips
        self.props = props

        self.stages = []
        self.stage_output_features = []
        self.stage_pool_kernel_size = []
        self.stage_conv_op_kernel_size = []

        assert len(pool_op_kernel_sizes) == len(conv_kernel_sizes)

        num_stages = len(conv_kernel_sizes)

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

        self.num_blocks_per_stage = num_blocks_per_stage  # decoder may need this
            
        if initial_conv:
            props_ini = deepcopy(props)
            props_ini['conv_op'] = props_ini['conv_op_base']
            self.initial_conv = CNDA(input_channels, base_num_features, (3,3,3), props_ini)
            current_input_features = base_num_features
        else:
            self.initial_conv = Identity()
            current_input_features = input_channels
            
        for stage in range(num_stages):
            current_output_features = min(int(base_num_features * feat_map_mul_on_downscale ** stage), max_num_features)
            current_kernel_size = conv_kernel_sizes[stage]
            current_pool_kernel_size = pool_op_kernel_sizes[stage]

            current_stage = StackedBlocks(current_input_features, current_output_features, current_kernel_size, props,
                                          num_blocks_per_stage[stage], num_convs_per_block[stage], block[stage], 
                                          current_pool_kernel_size)

            self.stages.append(current_stage)
            self.stage_output_features.append(current_output_features)
            self.stage_conv_op_kernel_size.append(current_kernel_size)
            self.stage_pool_kernel_size.append(current_pool_kernel_size)

            # update current_input_features
            current_input_features = current_output_features

        self.stages = nn.ModuleList(self.stages)
        self.output_features = current_output_features

    def forward(self, x, return_skips=None):
        """
        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        x = self.initial_conv(x)
        for s in self.stages:
            x = s(x)
            if self.default_return_skips:
                skips.append(x)

        if return_skips is None:
            return_skips = self.default_return_skips

        if return_skips:
            return skips
        else:
            return x

class GenericUNetDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, props=None, transposed_conv=True, 
                 block=StackedConvBlock, num_convs_per_block=2):
        super().__init__()
        self.num_classes = num_classes
        """
        We assume the bottleneck is part of the encoder, so we can start with upsample -> concat here
        """
        previous_stages = previous.stages
        previous_stage_output_features = previous.stage_output_features
        previous_stage_pool_kernel_size = previous.stage_pool_kernel_size
        previous_stage_conv_op_kernel_size = previous.stage_conv_op_kernel_size

        if props is None:
            self.props = previous.props
        else:
            self.props = props

        if self.props['conv_op_base'] == nn.Conv2d:
            transpconv = nn.ConvTranspose2d
            upsample_mode = "bilinear"
            resizeconv = ResizeConv2d
        elif self.props['conv_op_base'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
            resizeconv = ResizeConv3d
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.props['conv_op_base']))

        if num_blocks_per_stage is None:
            num_blocks_per_stage = previous.num_blocks_per_stage[:-1][::-1]

        assert len(num_blocks_per_stage) == len(previous.num_blocks_per_stage) - 1

        self.stage_pool_kernel_size = previous_stage_pool_kernel_size
        self.stage_output_features = previous_stage_output_features
        self.stage_conv_op_kernel_size = previous_stage_conv_op_kernel_size

        num_stages = len(previous_stages) - 1  # we have one less as the first stage here is what comes after the bottleneck

        if not isinstance(block, (list, tuple)):
            block = [block] * num_stages
        else:
            assert len(block) == num_stages
        if not isinstance(num_convs_per_block, (list, tuple)):
            num_convs_per_block = [num_convs_per_block] * num_stages
        else:
            assert len(num_convs_per_block) == num_stages
        
        self.tus = []
        self.stages = []

        for i, s in enumerate(np.arange(num_stages)[::-1]):
            features_below = previous_stage_output_features[s + 1]
            features_skip = previous_stage_output_features[s]

            if transposed_conv:
                self.tus.append(transpconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                                       previous_stage_pool_kernel_size[s + 1], bias=False))
            else:
                self.tus.append(resizeconv(features_below, features_skip, previous_stage_pool_kernel_size[s + 1],
                                       previous_stage_pool_kernel_size[s + 1], bias=False, mode=upsample_mode))
                
            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(StackedBlocks(2 * features_skip, features_skip, previous_stage_conv_op_kernel_size[s], 
                                             self.props, num_blocks_per_stage[i], num_convs_per_block[i], block[i]))

        self.segmentation_output = self.props['conv_op_base'](features_skip, num_classes, 1, 1, 0, 1, 1, False)

        self.tus = nn.ModuleList(self.tus)
        self.stages = nn.ModuleList(self.stages)

    def forward(self, skips, return_layers=False):
        # skips come from the encoder. They are sorted so that the bottleneck is last in the list
        # what is maybe not perfect is that the TUs and stages here are sorted the other way around
        # so let's just reverse the order of skips
        skips = skips[::-1]

        x = skips[0]  # this is the bottleneck
        
        layers = [x]
        for i in range(len(self.tus)):
            x = self.tus[i](x)
            x = torch.cat((x, skips[i + 1]), dim=1)
            x = self.stages[i](x)
            layers.append(x)

        segmentation = self.segmentation_output(x)
        layers = layers[::-1]

        if return_layers:
            return segmentation, layers
        else:
            return segmentation

class GenericUNet(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 max_features=512, transposed_conv=True, initial_conv=False, block_encoder=StackedConvBlock, block_decoder=StackedConvBlock, num_convs_per_block_encoder=2, num_convs_per_block_decoder=2,):
        
        super().__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = GenericUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                          feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes, props,
                                          initial_conv, max_features, block_encoder, num_convs_per_block_encoder, True)
        self.decoder = GenericUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props,
                                          transposed_conv, block_decoder, num_convs_per_block_decoder)

    def forward(self, x, return_layers=False):
        skips = self.encoder(x)
        return self.decoder(skips, return_layers)
