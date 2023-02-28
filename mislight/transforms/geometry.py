"""
Copied from monai.transforms.spatial, and only split randomize and __call__.

Fixed transform in __call__.
Randomize only with 'randomize' method.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from monai.data.meta_obj import get_track_meta
from monai.networks.layers import GaussianFilter
from monai.transforms import (
    Flip,
    RandAffineGrid,
    RandFlip,
    RandRotate90,
    Rand3DElastic,
    Resample, 
    Rotate90,
)
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import Randomizable, RandomizableTransform, Transform
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NdimageMode,
    NumpyPadMode,
    SplineMode,
    convert_to_cupy,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
)
from monai.utils.enums import GridPatchSort, PytorchPadMode, TraceKeys, TransformBackends, WSIPatchKeys

RandRange = Optional[Union[Sequence[Union[Tuple[float, float], float]], float]]


class FixRandRotate90(RandRotate90):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
        """
        if self._do_transform:
            out = Rotate90(self._rand_k, self.spatial_axes)(img)
        else:
            out = convert_to_tensor(img, track_meta=get_track_meta())

        if get_track_meta():
            maybe_rot90_info = self.pop_transform(out, check=False) if self._do_transform else {}
            self.push_transform(out, extra_info=maybe_rot90_info)
        return out
    
class FixRandFlip(RandFlip):
    """
    Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    Args:
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: channel first array, must have shape: (num_channels, H[, W, ..., ]),
            randomize: whether to execute `randomize()` function first, default to True.
        """
        out = self.flipper(img) if self._do_transform else img
        out = convert_to_tensor(out, track_meta=get_track_meta())
        if get_track_meta():
            xform_info = self.pop_transform(out, check=False) if self._do_transform else {}
            self.push_transform(out, extra_info=xform_info)
        return out
    
class FixRand3DElastic(Rand3DElastic):
    """
    Random elastic deformation and affine in 3D.
    A tutorial is available: https://github.com/Project-MONAI/tutorials/blob/0.6.0/modules/transforms_demo_2d.ipynb.
    """
    def randomize(self, img: Optional[torch.Tensor] = None) -> None:
        super(Rand3DElastic, self).randomize(None)
        if not self._do_transform:
            return None
        if img is None:
            return None
        grid_size = fall_back_tuple(self.spatial_size, img.shape[1:])
        self.rand_offset = self.R.uniform(-1.0, 1.0, [3] + list(grid_size)).astype(np.float32, copy=False)
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: torch.Tensor,
        spatial_size: tuple[int, int, int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W, D),
            spatial_size: specifying spatial 3D output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        """
        sp_size = fall_back_tuple(self.spatial_size, img.shape[1:])

        _device = img.device if isinstance(img, torch.Tensor) else self.device
        grid = create_grid(spatial_size=sp_size, device=_device, backend="torch")
        if self._do_transform:
            if self.rand_offset is None:
                raise RuntimeError("rand_offset is not initialized.")
            gaussian = GaussianFilter(3, self.sigma, 3.0).to(device=_device)
            offset = torch.as_tensor(self.rand_offset, device=_device).unsqueeze(0)
            grid[:3] += gaussian(offset)[0] * self.magnitude
            grid = self.rand_affine_grid(grid=grid)
        out: torch.Tensor = self.resampler(
            img,
            grid,  # type: ignore
            mode=mode if mode is not None else self.mode,
            padding_mode=padding_mode if padding_mode is not None else self.padding_mode,
        )
        return out