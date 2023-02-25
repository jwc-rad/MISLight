import torch
import torch.nn as nn
from typing import Optional, Sequence, Union

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock

from mislight.networks.blocks.cdtrans_block import CDTransformerBlock

class CDViT(nn.Module):
    """
    Re-implementation of CDTrans Vision Transformer.
    Based on "Xu et al., CDTrans: Cross-domain Transformer for Unsupervised Domain Adaptation <https://arxiv.org/abs/2109.06165> [https://github.com/CDTrans/CDTrans]".
    Adpated from monai.networks.nets.vit.ViT
    """
    def __init__(
        self,
        in_channels: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_layers: number of transformer blocks.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            classification: bool argument to determine if classification is used.
            num_classes: number of classes if classification is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dimensions.
            post_activation: add a final acivation function to the classification head when `classification` is True.
                Default to "Tanh" for `nn.Tanh()`. Set to other values to remove this function.
            qkv_bias: apply bias to the qkv linear layer in self attention block
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [CDTransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias) for i in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            if post_activation == "Tanh":
                self.classification_head = nn.Sequential(nn.Linear(hidden_size, num_classes), nn.Tanh())
            else:
                self.classification_head = nn.Linear(hidden_size, num_classes)  # type: ignore

    def forward(self, x, x2=None):
        if x2 is None:
            x = self.patch_embedding(x)
            if hasattr(self, "cls_token"):
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
            hidden_states_out = []
            for blk in self.blocks:
                x = blk(x)
                hidden_states_out.append(x)
            x = self.norm(x)
            if hasattr(self, "classification_head"):
                x = self.classification_head(x[:, 0])
            return x, hidden_states_out            
        else:
            x = self.patch_embedding(x)
            x2 = self.patch_embedding(x2)
            if hasattr(self, "cls_token"):
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                cls_token = self.cls_token.expand(x2.shape[0], -1, -1)
                x2 = torch.cat((cls_token, x2), dim=1)
            x3 = x2
            hidden_states_out = []
            hidden_states_out2 = []
            hidden_states_out3 = []
            for blk in self.blocks:
                x, x2, x3 = blk(x, x2, x3)
                hidden_states_out.append(x)
                hidden_states_out2.append(x2)
                hidden_states_out3.append(x3)
            x = self.norm(x)
            x2 = self.norm(x2)
            x3 = self.norm(x3)
            if hasattr(self, "classification_head"):
                x = self.classification_head(x[:, 0])
                x2 = self.classification_head(x2[:, 0])
                x3 = self.classification_head(x3[:, 0])
            return x, x2, x3, hidden_states_out, hidden_states_out2, hidden_states_out3