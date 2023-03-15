from functools import partial
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from monai.networks.layers import get_act_layer, DropPath
from monai.networks.layers.factories import Conv, Norm, split_args
from monai.utils import ensure_tuple_rep, optional_import, has_option

rearrange, _ = optional_import("einops", name="rearrange")
Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

def get_norm_layer(name: Union[tuple, str], spatial_dims: Optional[int] = 1, channels: Optional[int] = 1):
    if name == "":
        return torch.nn.Identity()
    norm_name, norm_args = split_args(name)
    norm_type = Norm[norm_name, spatial_dims]
    kw_args = dict(norm_args)
    if has_option(norm_type, "num_features") and "num_features" not in kw_args:
        kw_args["num_features"] = channels
    if has_option(norm_type, "num_channels") and "num_channels" not in kw_args:
        kw_args["num_channels"] = channels
    return partial(norm_type, **kw_args)

class MixFFN(nn.Module):
    """
    Mix-FFN, based on "Xie et al., 
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers <https://arxiv.org/abs/2105.15203>"
    Re-implemented from https://github.com/NVlabs/SegFormer/mmseg/models/backbones/mix_transformer.py, https://github.com/open-mmlab/mmsegmentation/mmseg/models/backbones/mit.py
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        hidden_size: int,
        dropout_rate: float = 0.0,
        act: Union[tuple, str] = 'GELU',
    ):
        super().__init__()
        assert spatial_dims in [2, 3]
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        self.fc1 = Conv[Conv.CONV, spatial_dims](in_channels, hidden_size, kernel_size=1, stride=1, bias=True)
        self.pe_conv = Conv[Conv.CONV, spatial_dims](hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_size)
        self.fc2 = Conv[Conv.CONV, spatial_dims](hidden_size, in_channels, kernel_size=1, stride=1, bias=True)
        self.drop = nn.Dropout(dropout_rate)
        self.act = get_act_layer(act)
        
    def forward(self, x, x_shape):
        if self.spatial_dims == 2:
            h, w = x_shape
            x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        else:
            h, w, d = x_shape
            x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d)
        
        x = self.fc1(x)
        x = self.pe_conv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        if self.spatial_dims == 2:
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            x = rearrange(x, "b c h w d -> b (h w d) c")
        return x
    
class EfficientSABlock(nn.Module):
    """
    Efficient Self-Attention Block, based on: "Xie et al., 
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers <https://arxiv.org/abs/2105.15203>"
    Re-implemented from https://github.com/NVlabs/SegFormer/mmseg/models/backbones/mix_transformer.py, https://github.com/open-mmlab/mmsegmentation/mmseg/models/backbones/mit.py
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        qkv_bias: bool = False,
        sr_ratio: int = 1,
        spatial_dims: int = 2,
    ):
        super().__init__()    
        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")
        assert spatial_dims in [2, 3]
        self.spatial_dims = spatial_dims
        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)        
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.input_rearrange1 = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=1, l=num_heads)
        self.input_rearrange2 = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=2, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(drop)
        self.drop_weights = nn.Dropout(attn_drop)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv[Conv.CONV, spatial_dims](hidden_size, hidden_size, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, x_shape):
        output = self.input_rearrange1(self.q(x))
        q = output[0]
        x_kv = x
        if self.sr_ratio > 1:
            if self.spatial_dims == 2:
                h, w = x_shape
                x_kv = rearrange(x_kv, "b (h w) c -> b c h w", h=h, w=w)
            else:
                h, w, d = x_shape
                x_kv = rearrange(x_kv, "b (h w d) c -> b c h w d", h=h, w=w, d=d)
            x_kv = self.sr(x_kv)
            if self.spatial_dims == 2:
                x_kv = rearrange(x_kv, "b c h w -> b (h w) c")
            else:
                x_kv = rearrange(x_kv, "b c h w d -> b (h w d) c")
        output = self.input_rearrange2(self.kv(x_kv))
        k, v = output[0], output[1]
        
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = self.out_rearrange(x)
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x
    

class MixTransformerLayer(nn.Module):
    """
    One encoder layer  in Mix Transformer, based on: "Xie et al., 
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers <https://arxiv.org/abs/2105.15203>"
    Re-implemented from https://github.com/NVlabs/SegFormer/mmseg/models/backbones/mix_transformer.py, https://github.com/open-mmlab/mmsegmentation/mmseg/models/backbones/mit.py
    """

    def __init__(
        self, 
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Union[tuple, str] = 'GELU',
        norm_layer: Union[tuple, str] = 'layer',
        sr_ratio: int = 1,
        spatial_dims: int = 2,
    ):
        super().__init__()
        norm_layer = get_norm_layer(norm_layer, spatial_dims)
        self.norm1 = norm_layer(dim)
        self.attn = EfficientSABlock(dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, drop=drop, sr_ratio=sr_ratio, spatial_dims=spatial_dims)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(
            spatial_dims,
            dim,
            mlp_dim,
            dropout_rate=drop,
            act=act_layer,
        )
        
    def forward(self, x, x_shape):
        x = x + self.drop_path(self.attn(self.norm1(x), x_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x), x_shape))
        return x
    
class OverlapPatchEmbed(nn.Module):
    """
    Overlapped Patching Embedding, based on: "Xie et al., 
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers <https://arxiv.org/abs/2105.15203>"
    Re-implemented from https://github.com/NVlabs/SegFormer/mmseg/models/backbones/mix_transformer.py
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        hidden_size: int,
        padding: Optional[Union[Sequence[int], int]] = None,
        norm_layer: Optional[Union[tuple, str]] = 'layer',
        spatial_dims: int = 2,
    ):
        super().__init__()
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        stride = ensure_tuple_rep(stride, spatial_dims)
        if padding is None:
            padding = tuple([i//2 for i in patch_size])
        else:
            padding = ensure_tuple_rep(padding, spatial_dims)
        self.spatial_dims = spatial_dims

        self.proj = Conv[Conv.CONV, spatial_dims](
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        
        if norm_layer is None:
            self.norm = None
        else:
            self.norm = get_norm_layer(norm_layer, spatial_dims)(hidden_size)

    def forward(self, x):
        x = self.proj(x)
        x_shape = x.shape[2:]
        
        if self.spatial_dims == 2:
            x = rearrange(x, "b c h w -> b (h w) c")
        else:
            x = rearrange(x, "b c h w d -> b (h w d) c")
        
        if self.norm is not None:
            x = self.norm(x)

        return x, x_shape
    

class MixVisionTransformer(nn.Module):
    """
    Mix Vision Transformer Encoder in SegFormer, based on: "Xie et al., 
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers <https://arxiv.org/abs/2105.15203>"
    Re-implemented from https://github.com/NVlabs/SegFormer/mmseg/models/backbones/mix_transformer.py, https://github.com/open-mmlab/mmsegmentation/mmseg/models/backbones/mit.py
    """    
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        num_stages: int = 4,
        num_layers: Union[Sequence[int], int] = [2,2,2,2],
        num_heads: Union[Sequence[int], int] = [1,2,5,8],
        patch_sizes: Union[Sequence[Sequence[int]], Sequence[int], int] = [7,3,3,3],
        strides: Union[Sequence[Sequence[int]], Sequence[int], int] = [4,2,2,2],
        paddings: Optional[Union[Sequence[int], int]] = None,
        sr_ratios: Union[Sequence[int], int] = [8,4,2,1],
        embed_dims: Union[Sequence[int], int] = 32,
        mlp_ratios: Union[Sequence[int], int] = 4,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.1,
        act_layer: Union[tuple, str] = 'GELU',
        norm_layer: Union[tuple, str] = 'layer',
        out_indices: Union[Sequence[int], int] = [0, 1, 2, 3],
    ):
        super().__init__()
    
        self.spatial_dims = spatial_dims
    
        num_layers = ensure_tuple_rep(num_layers, num_stages)
        num_heads = ensure_tuple_rep(num_heads, num_stages)
        patch_sizes = ensure_tuple_rep(patch_sizes, num_stages)
        strides = ensure_tuple_rep(strides, num_stages)
        paddings = ensure_tuple_rep(paddings, num_stages)
        sr_ratios = ensure_tuple_rep(sr_ratios, num_stages)
        embed_dims = ensure_tuple_rep(embed_dims, num_stages)
        mlp_ratios = ensure_tuple_rep(mlp_ratios, num_stages)
    
        self.out_indices = out_indices
        if isinstance(out_indices, Sequence):
            assert max(out_indices) < num_stages
        else:
            assert out_indices < num_stages
    
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path, sum(num_layers))
        ]  # stochastic num_layer decay rule
    
        cur = 0
        self.layers = nn.ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims[i] * num_heads[i]            
            patch_embed = OverlapPatchEmbed(
                in_channels=in_channels,
                hidden_size=embed_dims_i,
                patch_size=patch_sizes[i],
                stride=strides[i],
                padding=paddings[i],
                norm_layer=norm_layer,
                spatial_dims=spatial_dims,
            )
            layer = nn.ModuleList([
                MixTransformerLayer(
                    dim=embed_dims_i,
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=dpr[cur + idx],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i],
                    spatial_dims=spatial_dims,
                ) for idx in range(num_layer)
            ])
            norm = Norm[norm_layer, spatial_dims](embed_dims_i)
            self.layers.append(nn.ModuleList([patch_embed, layer, norm]))
            
            in_channels = embed_dims_i
            cur += num_layer
            
    def forward(self, x):
        outs = []

        for i, layer in enumerate(self.layers):
            x, x_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, x_shape)
            x = layer[2](x)

            if self.spatial_dims == 2:
                h, w = x_shape
                x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
            else:
                h, w, d = x_shape
                x = rearrange(x, "b (h w d) c -> b c h w d", h=h, w=w, d=d)                

            if isinstance(self.out_indices, Sequence):    
                if i in self.out_indices:
                    outs.append(x)
            else:
                if i == self.out_indices:
                    return x

        return outs