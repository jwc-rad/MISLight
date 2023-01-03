'''Copied with little changes from https://github.com/taesungp/contrastive-unpaired-translation/models/networks.py
'''

import functools
import numpy as np
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from mislight.networks.utils import init_weights
from mislight.utils.misc import safe_repeat

class PatchSampleF(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        channels: Union[Sequence[int], int], 
        use_mlp: bool = False, 
        init_type: str = 'xavier',
    ):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        self.spatial_dims = spatial_dims
        self.nc = safe_repeat(channels, 1)[0]  # hard-coded
        self.use_mlp = use_mlp
        self.init_type = init_type
        
        self.l2norm = functools.partial(F.normalize, p=2, dim=1)
        self.mlp_init = False

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_weights(self, self.init_type)
        self.to(feats[0].device)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B = feat.shape[0]
            SPATIAL = feat.shape[2:]
            feat_reshape = feat.permute(*([0]+[i for i in range(2,2+self.spatial_dims)]+[1])).flatten(*([i for i in range(1,1+self.spatial_dims)]))
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1]] + list(SPATIAL))
            return_feats.append(x_sample)
        return return_feats, return_ids
    
class PoolingF(nn.Module):
    def __init__(self):
        super().__init__()
        model = [nn.AdaptiveMaxPool2d(1)]
        self.model = nn.Sequential(*model)
        self.l2norm = functools.partial(F.normalize, p=2, dim=1)

    def forward(self, x):
        return self.l2norm(self.model(x))

class ReshapeF(nn.Module):
    def __init__(self):
        super().__init__()
        model = [nn.AdaptiveAvgPool2d(4)]
        self.model = nn.Sequential(*model)
        self.l2norm = functools.partial(F.normalize, p=2, dim=1)

    def forward(self, x):
        x = self.model(x)
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)
        return self.l2norm(x_reshape)