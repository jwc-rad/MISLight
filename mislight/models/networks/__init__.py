import numpy as np
from .blocks import StackedConvBlock, StackedConvResidualBlock
from .unet import GenericUNet
from .attentionunet import GenericAttentionUNet
from .usenet import GenericUSENet
from .utils import init_weights, get_default_network_config
from .layers import DepthwiseSeparableConv3d

def define_S(nc_input, num_classes, netS='unet', nsf=16, n_stages=5, n_blocks_per_stage=1, max_features=320, transposed_conv=True, nonlin='LeakyReLU', norm='in', init_type='xavier', opt=None):
    '''
    Args:
        nc_input: # of input channels
        num_classes: # of positive classes
        netS: name of network
        nsf: # of base features
        n_stages: # of stages of network, linked to # of downsampling/upsampling
        n_blocks_per_stage
        max_features: # of features in block is min(nsf * 2**stages, max_features)
        transposed_conv: if True, transposed convolution for upsampling.
        nonlin: nonlinear activation
        norm: in=instance normalization, bn=batch normalization
        init_type: e.g. xavier, kaiming
    '''
    net = None
    
    num_blocks_per_stage_encoder = (n_blocks_per_stage,)*n_stages
    num_blocks_per_stage_decoder = (n_blocks_per_stage,)*(n_stages-1)
    conv_op_kernel_sizes = ((3,3,3),)*n_stages
    pool_op_kernel_sizes = ((1,1,1),) + ((2,2,2),)*(n_stages-1)
    props = get_default_network_config(dim=3, dropout_p=None, nonlin=nonlin, norm_type=norm) 
    
    parse_net = netS.split('_')
    
    if 'mobile' in parse_net:
        props.update(conv_op=DepthwiseSeparableConv3d)
    
    generic = GenericUNet
    if 'attention' in parse_net:
        generic = GenericAttentionUNet
    elif 'se' in parse_net:
        generic = GenericUSENet
    
    if 'unet' in parse_net:
        net = generic(nc_input, nsf, num_blocks_per_stage_encoder, 2, pool_op_kernel_sizes, conv_op_kernel_sizes, props, 
                      num_classes+1, num_blocks_per_stage_decoder, max_features, transposed_conv,
                      initial_conv=False, block_encoder=StackedConvBlock, block_decoder=StackedConvBlock,
                      num_convs_per_block_encoder=2, num_convs_per_block_decoder=2)
    elif 'resunet' in parse_net:
        '''adapted from FabiansUNet of nnUNet (https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_modular_residual_UNet.py)
        '''
        num_blocks_per_stage_encoder = tuple(np.arange(n_stages)+1)
        num_blocks_per_stage_decoder = (1,)*(n_stages-1)
        num_convs_per_block_decoder = 2
        
        net = generic(nc_input, nsf, num_blocks_per_stage_encoder, 2, pool_op_kernel_sizes, conv_op_kernel_sizes, props, 
                      num_classes+1, num_blocks_per_stage_decoder, max_features, transposed_conv,
                      initial_conv=True, block_encoder=StackedConvResidualBlock, block_decoder=StackedConvBlock,
                      num_convs_per_block_encoder=2, num_convs_per_block_decoder=num_convs_per_block_decoder)        
             
    return init_weights(net, init_type)
