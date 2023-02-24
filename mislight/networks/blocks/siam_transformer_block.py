import torch
import torch.nn as nn
from typing import Optional, Sequence, Union

from monai.networks.blocks.mlp import MLPBlock
from monai.utils import optional_import

Rearrange, _ = optional_import("einops.layers.torch", name="Rearrange")

class SiamAttnBlock(nn.Module):
    """
    Adapted from monai.networks.blocks.selfattention.SABlock
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: bias term for the qkv linear layer.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

    def forward(self, x, x2=None):
        output = self.input_rearrange(self.qkv(x))
        q, k, v = output[0], output[1], output[2]
        if x2 is not None:
            output = self.input_rearrange(self.qkv(x2))
            q2, k2, v2 = output[0], output[1], output[2]
        
            q = torch.cat([q, q2], 2)
            k = torch.cat([k, k2], 2)
            v = torch.cat([v, v2], 2)
        
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        
        if x2 is not None:
            x, x2 = x.chunk(2, dim=2)

            x = self.out_rearrange(x)
            x = self.out_proj(x)
            x = self.drop_output(x)
            x2 = self.out_rearrange(x2)
            x2 = self.out_proj(x2)
            x2 = self.drop_output(x2)
            return x, x2
        else:
            x = self.out_rearrange(x)
            x = self.out_proj(x)
            x = self.drop_output(x)
            return x
        
        
class SiamTransformerBlock(nn.Module):
    """
    Adapted from monai.networks.blocks.transformerblock.TransformerBlock
    """
    def __init__(
        self, hidden_size: int, mlp_dim: int, num_heads: int, dropout_rate: float = 0.0, qkv_bias: bool = False
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            qkv_bias: apply bias term for the qkv linear layer
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SiamAttnBlock(hidden_size, num_heads, dropout_rate, qkv_bias)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, x2=None):
        if x2 is None:        
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x
        else:
            a, a2 = self.attn(self.norm1(x), self.norm1(x2))
            x = x + a
            x = x + self.mlp(self.norm2(x))
            x2 = x2 + a2
            x2 = x2 + self.mlp(self.norm2(x2))
            return x, x2