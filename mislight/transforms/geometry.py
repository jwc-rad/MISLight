"""
Copied from monai.transforms.spatial, and only split randomize and __call__.

Fixed transform in __call__.
Randomize only with 'randomize' method.
"""

import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from monai.data.meta_obj import get_track_meta
from monai.networks.layers import GaussianFilter
from monai.transforms import (
    Flip,
    RandAffineGrid,
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

class FixRandRotate90(RandomizableTransform, InvertibleTransform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    backend = Rotate90.backend

    def __init__(self, prob: float = 0.1, max_k: int = 3, spatial_axes: tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`, (Default 3).
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        RandomizableTransform.__init__(self, prob)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._rand_k = 0

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self._rand_k = self.R.randint(self.max_k) + 1

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

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        xform_info = self.pop_transform(data)
        if not xform_info[TraceKeys.DO_TRANSFORM]:
            return data
        rotate_xform = xform_info[TraceKeys.EXTRA_INFO]
        return Rotate90().inverse_transform(data, rotate_xform)
    
class FixRandFlip(RandomizableTransform, InvertibleTransform):
    """
    Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html
    Args:
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """

    backend = Flip.backend

    def __init__(self, prob: float = 0.1, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        RandomizableTransform.__init__(self, prob)
        self.flipper = Flip(spatial_axis=spatial_axis)

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

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        transform = self.pop_transform(data)
        if not transform[TraceKeys.DO_TRANSFORM]:
            return data
        data.applied_operations.append(transform[TraceKeys.EXTRA_INFO])  # type: ignore
        return self.flipper.inverse(data)