'''
USE-Net: incorporating Squeeze-and-Excitation blocks into U-Net for prostate zonal segmentation of multi-institutional MRI datasets (https://arxiv.org/abs/1904.08254)
'''

from copy import deepcopy
import numpy as np
import torch
from torch import nn

from monai.networks.blocks.squeeze_and_excitation import ResidualSELayer

from .layers import Identity, ResizeConv3d, ResizeConv2d
from .blocks import CNDA, StackedBlocks, StackedConvBlock, StackedConvResidualBlock
from .unet import GenericUNetEncoder

class GenericUSENetDecoder(nn.Module):
    def __init__(self, previous, num_classes, num_blocks_per_stage=None, props=None, transposed_conv=True, 
                 block=StackedConvBlock, num_convs_per_block=2, reduction=8, se_encoder=True, se_decoder=True):
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
            spatial_dims = 2
        elif self.props['conv_op_base'] == nn.Conv3d:
            transpconv = nn.ConvTranspose3d
            upsample_mode = "trilinear"
            resizeconv = ResizeConv3d
            spatial_dims = 3
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
        self.ses = []
        self.se_encoder = se_encoder
        self.se_decoder = se_decoder
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
                
            self.ses.append(ResidualSELayer(spatial_dims, features_skip, reduction))
                
            # after we tu we concat features so now we have 2xfeatures_skip
            self.stages.append(StackedBlocks(2 * features_skip, features_skip, previous_stage_conv_op_kernel_size[s], 
                                             self.props, num_blocks_per_stage[i], num_convs_per_block[i], block[i]))

        self.segmentation_output = self.props['conv_op_base'](features_skip, num_classes, 1, 1, 0, 1, 1, False)

        self.tus = nn.ModuleList(self.tus)
        self.ses = nn.ModuleList(self.ses)
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
            if self.se_decoder:
                x = self.ses[i](x)
            current_skip = skips[i+1]
            if self.se_encoder:
                current_skip = self.ses[i](current_skip)
            x = torch.cat((x, current_skip), dim=1)
            x = self.stages[i](x)
            layers.append(x)

        segmentation = self.segmentation_output(x)
        layers = layers[::-1]

        if return_layers:
            return segmentation, layers
        else:
            return segmentation
        
    
class GenericUSENet(nn.Module):
    def __init__(self, input_channels, base_num_features, num_blocks_per_stage_encoder, feat_map_mul_on_downscale,
                 pool_op_kernel_sizes, conv_kernel_sizes, props, num_classes, num_blocks_per_stage_decoder,
                 max_features=512, transposed_conv=True, initial_conv=False, block_encoder=StackedConvBlock, block_decoder=StackedConvBlock, num_convs_per_block_encoder=2, num_convs_per_block_decoder=2,
                 reduction=8, se_encoder=True, se_decoder=True):
        
        super().__init__()
        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = GenericUNetEncoder(input_channels, base_num_features, num_blocks_per_stage_encoder,
                                          feat_map_mul_on_downscale, pool_op_kernel_sizes, conv_kernel_sizes, props,
                                          initial_conv, max_features, block_encoder, num_convs_per_block_encoder, True)
        self.decoder = GenericUSENetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props,
                                            transposed_conv, block_decoder, num_convs_per_block_decoder, 
                                            reduction, se_encoder, se_decoder)

    def forward(self, x, return_layers=False):
        skips = self.encoder(x)
        return self.decoder(skips, return_layers)
