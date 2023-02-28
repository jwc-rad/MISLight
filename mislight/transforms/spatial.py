import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from monai.data.meta_obj import get_track_meta
from monai.networks.layers import GaussianFilter
from monai.transforms import (
    RandAffineGrid,
    Resample, 
)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import Randomizable, RandomizableTransform, Transform
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    fall_back_tuple,
)

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]

# modified from monai.transforms.Rand3DElastic
class RandElasticGrid(RandomizableTransform):
    backend = Resample.backend
    
    def __init__(
        self,
        spatial_dims: int,
        sigma_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
        prob: float = 0.1,
        rotate_range: RandRange = None,
        shear_range: RandRange = None,
        translate_range: RandRange = None,
        scale_range: RandRange = None,
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        mode: Union[str, int] = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        device: Optional[torch.device] = None,
    ) -> None:
        RandomizableTransform.__init__(self, prob)
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            device=device,
        )
        self.resampler = Resample(device=device)

        self.spatial_dims = spatial_dims
        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode: str = padding_mode
        self.device = device

        self.rand_offset: np.ndarray
        self.magnitude = 1.0
        self.sigma = 1.0


    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ):
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    
    def set_device(self, device):
        self.rand_affine_grid.device = device
        self.resampler.device = device
        self.device = device

    def randomize(self, grid_size: Sequence[int]) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.rand_offset = self.R.uniform(-1.0, 1.0, [self.spatial_dims] + list(grid_size)).astype(np.float32, copy=False)
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        mode: Union[str, int, None] = None,
        padding_mode: Optional[str] = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        if randomize:
            self.randomize(grid_size=sp_size)

        _device = img.device if isinstance(img, torch.Tensor) else self.device
        grid = create_grid(spatial_size=sp_size, device=_device, backend="torch")
        if self._do_transform:
            if self.rand_offset is None:
                raise RuntimeError("rand_offset is not initialized.")
            gaussian = GaussianFilter(self.spatial_dims, self.sigma, 3.0).to(device=_device)
            offset = torch.as_tensor(self.rand_offset, device=_device).unsqueeze(0)
            grid[:self.spatial_dims] += gaussian(offset)[0] * self.magnitude
            grid = self.rand_affine_grid(grid=grid)
        
        return grid
    
    def resample(
        self,
        img: torch.Tensor,
        grid: torch.Tensor,
        mode: Union[str, int, None] = None,
        padding_mode: Optional[str] = None,        
    ) -> torch.Tensor:
        out: torch.Tensor = self.resampler(
            img,
            grid,  # type: ignore
            mode=mode if mode is not None else self.mode,
            padding_mode=padding_mode if padding_mode is not None else self.padding_mode,
        )
        return out
    
