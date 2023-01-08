import copy
import itertools
from itertools import chain
from math import ceil
import numbers
import numpy as np
import random
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F

from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.transforms import (
    BorderPad,
    CenterSpatialCrop,
    Compose,
    Crop,
    Cropd,
    Transform,
    MapTransform,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    GridPatchd,
    Flip,
    InvertibleTransform,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    Pad,
    RandCropd,
    RandFlipd,
    RandGridPatchd,
    RandCropd,
    Randomizable,
    RandRotate90d,
    RandSpatialCrop,
    Resize,
    ScaleIntensityRanged,
    SpatialPad,
    SplitDimd,
    ToTensord,
)

from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.utils import (
    compute_divisible_spatial_size,
    convert_pad_mode,
    create_translate,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    generate_spatial_bounding_box,
    is_positive,
    map_binary_to_indices,
    map_classes_to_indices,
    weighted_patch_samples,
)
from monai.transforms.utils_pytorch_numpy_unification import clip as clip_torch_np
from monai.transforms.utils_pytorch_numpy_unification import (
    any_np_pt,
    ascontiguousarray,
    cumsum,
    isfinite,
    nonzero,
    ravel,
    searchsorted,
    unique,
    unravel_index,
    where,
)
from monai.utils import (
    Method,
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    PytorchPadMode,
    TraceKeys,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    issequenceiterable,
    optional_import,
    pytorch_after,
)
from monai.utils.enums import GridPatchSort, TransformBackends
from monai.utils.module import look_up_option
from monai.utils.type_conversion import convert_data_type

class ConvertLabel(Transform):
    def __init__(self, convert_dict: dict = {}) -> None:
        self.convert_dict = convert_dict
        
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if len(self.convert_dict.keys()) > 0:
            select_k = []
            select_v = []
            for k,v in self.convert_dict.items():
                select_k.append(img==k)
                select_v.append(v)
            img = np.select(select_k, select_v, img)
            return img
        else:
            return img
        
class ConvertLabeld(MapTransform):
    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = ConvertLabel(*args, **kwargs)
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.__call__(d[key])
        return d 

class RandHorizontalFlipLabeld(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, prob=0.1, convert_dict=None) -> None:
        super().__init__(keys)
        self.prob = prob
        self.flipper = Flip(spatial_axis=0)
        if convert_dict is None:
            self.convert_label = None
        else:
            self.convert_label = ConvertLabel(convert_dict)
        self._do_transform = False        
        
    def randomize(self):
        self._do_transform = self.R.random_sample() < self.prob        
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        self.randomize()
        d = dict(data)
        for key in self.keys:
            if self._do_transform:
                d[key] = self.flipper(d[key])
                if key == 'mask' and not (self.convert_label is None):
                    d[key] = self.convert_label(d[key])
        return d 
    
class ResizeV2(Resize):
    '''MONAI's Resize Extension
    Changes:
        additional size_mode "shortest"
        clip
    '''

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        size_mode: str = "all",
        mode: Union[InterpolateMode, str] = InterpolateMode.AREA,
        align_corners: Optional[bool] = None,
        anti_aliasing: bool = False,
        anti_aliasing_sigma: Union[Sequence[float], float, None] = None,
        clip: bool = False,
    ) -> None:
        self.size_mode = look_up_option(size_mode, ["all", "longest", "shortest"])
        self.spatial_size = spatial_size
        self.mode: InterpolateMode = look_up_option(mode, InterpolateMode)
        self.align_corners = align_corners
        self.anti_aliasing = anti_aliasing
        self.anti_aliasing_sigma = anti_aliasing_sigma
        self.clip = clip

    def __call__(
        self,
        img: NdarrayOrTensor,
        mode: Optional[Union[InterpolateMode, str]] = None,
        align_corners: Optional[bool] = None,
        anti_aliasing: Optional[bool] = None,
        anti_aliasing_sigma: Union[Sequence[float], float, None] = None,
        spatial_size: Union[Sequence[int], int, None] = None,
    ) -> NdarrayOrTensor:
        anti_aliasing = self.anti_aliasing if anti_aliasing is None else anti_aliasing
        anti_aliasing_sigma = self.anti_aliasing_sigma if anti_aliasing_sigma is None else anti_aliasing_sigma
        spatial_size_call = self.spatial_size if spatial_size is None else spatial_size
        
        if self.size_mode == "all":
            input_ndim = img.ndim - 1  # spatial ndim
            output_ndim = len(ensure_tuple(spatial_size_call))
            if output_ndim > input_ndim:
                input_shape = ensure_tuple_size(img.shape, output_ndim + 1, 1)
                img = img.reshape(input_shape)
            elif output_ndim < input_ndim:
                raise ValueError(
                    "len(spatial_size) must be greater or equal to img spatial dimensions, "
                    f"got spatial_size={output_ndim} img={input_ndim}."
                )
            spatial_size_ = fall_back_tuple(spatial_size_call, img.shape[1:])
        elif self.size_mode == "longest":  # for the "longest" mode
            img_size = img.shape[1:]
            if not isinstance(spatial_size_call, int):
                raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
            scale = spatial_size_call / max(img_size)
            spatial_size_ = tuple(int(round(s * scale)) for s in img_size)
        else:  # for the "shortest" mode
            img_size = img.shape[1:]
            if not isinstance(spatial_size_call, int):
                raise ValueError("spatial_size must be an int number if size_mode is 'shortest'.")
            scale = spatial_size_call / min(img_size)
            spatial_size_ = tuple(int(round(s * scale)) for s in img_size)

        if tuple(img.shape[1:]) == spatial_size_:  # spatial shape is already the desired
            return img
        img_, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        if anti_aliasing and any(x < y for x, y in zip(spatial_size_, img_.shape[1:])):
            factors = torch.div(torch.Tensor(list(img_.shape[1:])), torch.Tensor(spatial_size_))
            if anti_aliasing_sigma is None:
                # if sigma is not given, use the default sigma in skimage.transform.resize
                anti_aliasing_sigma = torch.maximum(torch.zeros(factors.shape), (factors - 1) / 2).tolist()
            else:
                # if sigma is given, use the given value for downsampling axis
                anti_aliasing_sigma = list(ensure_tuple_rep(anti_aliasing_sigma, len(spatial_size_)))
                for axis in range(len(spatial_size_)):
                    anti_aliasing_sigma[axis] = anti_aliasing_sigma[axis] * int(factors[axis] > 1)
            anti_aliasing_filter = GaussianSmooth(sigma=anti_aliasing_sigma)
            img_ = anti_aliasing_filter(img_)

        resized = torch.nn.functional.interpolate(
            input=img_.unsqueeze(0),
            size=spatial_size_,
            mode=look_up_option(self.mode if mode is None else mode, InterpolateMode).value,
            align_corners=self.align_corners if align_corners is None else align_corners,
        )
        out, *_ = convert_to_dst_type(resized.squeeze(0), img)
        
        if self.clip:
            i_min = img.min()
            i_max = img.max()
            out = clip_torch_np(out, i_min, i_max)
        return out
    
class ResizeV2d(MapTransform, InvertibleTransform):
    
    backend = ResizeV2.backend
    
    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        size_mode: str = "all",
        mode: SequenceStr = InterpolateMode.AREA,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        anti_aliasing: Union[Sequence[bool], bool] = False,
        anti_aliasing_sigma: Union[Sequence[Union[Sequence[float], float, None]], Sequence[float], float, None] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.anti_aliasing = ensure_tuple_rep(anti_aliasing, len(self.keys))
        self.anti_aliasing_sigma = ensure_tuple_rep(anti_aliasing_sigma, len(self.keys))
        self.resizer = ResizeV2(spatial_size=spatial_size, size_mode=size_mode)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key, mode, align_corners, anti_aliasing, anti_aliasing_sigma in self.key_iterator(
            d, self.mode, self.align_corners, self.anti_aliasing, self.anti_aliasing_sigma
        ):
            d[key] = self.resizer(
                d[key],
                mode=mode,
                align_corners=align_corners,
                anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma,
            )
        return d

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.resizer.inverse(d[key])
        return d
    
class RandResizeV2(Randomizable, ResizeV2):
    '''Random resizing
    if max_spatial_size is None, just normal Resize
    '''

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        *args,
        max_spatial_size: Optional[Union[Sequence[int], int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_spatial_size = max_spatial_size

    def randomize(self) -> None:
        spatial_size = self.spatial_size
        if isinstance(self.spatial_size, int):
            spatial_size = (spatial_size,)
        max_size = spatial_size if self.max_spatial_size is None else fall_back_tuple(self.max_spatial_size, spatial_size)
        
        if any(i > j for i, j in zip(spatial_size, max_size)):
            raise ValueError(f"min ROI size: {spatial_size} is larger than max ROI size: {max_size}.")
        self._size = tuple(self.R.randint(low=spatial_size[i], high=max_size[i] + 1) for i in range(len(spatial_size)))
        if isinstance(self.spatial_size, int):
            self._size = self._size[0] 
        
    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:  # type: ignore
        if randomize:
            self.randomize()
        if self._size is None:
            raise RuntimeError("self._size not specified.")

        return super().__call__(img, spatial_size=self._size)
    
class RandResizeV2d(Randomizable, ResizeV2d):
    
    backend = ResizeV2.backend
    
    def __init__(
        self,
        *args,
        max_spatial_size: Optional[Union[Sequence[int], int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.max_spatial_size = max_spatial_size

    def randomize(self) -> None:
        spatial_size = self.resizer.spatial_size
        if isinstance(self.resizer.spatial_size, int):
            spatial_size = (spatial_size,)
        max_size = spatial_size if self.max_spatial_size is None else fall_back_tuple(self.max_spatial_size, spatial_size)
        if any(i > j for i, j in zip(spatial_size, max_size)):
            raise ValueError(f"min ROI size: {spatial_size} is larger than max ROI size: {max_size}.")
        self._size = tuple(self.R.randint(low=spatial_size[i], high=max_size[i] + 1) for i in range(len(spatial_size)))
        if isinstance(self.resizer.spatial_size, int):
            self._size = self._size[0] 

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        not_randomized = True
        d = dict(data)
        for key, mode, align_corners, anti_aliasing, anti_aliasing_sigma in self.key_iterator(
            d, self.mode, self.align_corners, self.anti_aliasing, self.anti_aliasing_sigma
        ):
            if not_randomized:
                self.randomize()
                not_randomized = False
            if self._size is None:
                raise RuntimeError("self.resizer._size not specified.")
                
            d[key] = self.resizer(
                d[key],
                mode=mode,
                align_corners=align_corners,
                anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma,
                spatial_size=self._size,
            )
        return d
    
    
class RandSpatialPadCrop(Randomizable, Crop):
    """
    Extension of MONAI's RandSpatialCrop
    Allow ROI size larger than input image size
    Args:
        pad_tolerance: 1 means padding to ROI_size, >1 mean larger than ROI_size. should not be less than 1
    """
    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        max_roi_size: Optional[Union[Sequence[int], int]] = None,
        random_center: bool = True,
        pad_tolerance: float = 1,
        method: str = Method.SYMMETRIC,
        mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ) -> None:
        self.roi_size = roi_size
        self.max_roi_size = max_roi_size
        self.random_center = random_center
        self.pad_tolerance = pad_tolerance
        self.method = method
        self.mode = mode
        self.pad_kwargs = pad_kwargs
        self._size: Optional[Sequence[int]] = None
        self._slices: Tuple[slice, ...]

    def randomize(self, img_size: Sequence[int]) -> None:
        self._size = fall_back_tuple(self.roi_size, img_size)
        max_size = self._size if self.max_roi_size is None else fall_back_tuple(self.max_roi_size, img_size)
        if any(i > j for i, j in zip(self._size, max_size)):
            raise ValueError(f"min ROI size: {self._size} is larger than max ROI size: {max_size}.")
        self._size = tuple(self.R.randint(low=self._size[i], high=max_size[i] + 1) for i in range(len(img_size)))
            
        img_size = tuple(j + int(self.pad_tolerance*max(0, i-j)) for i,j in zip(self._size, img_size))
        
        self._padder = SpatialPad(spatial_size=img_size, method=self.method, mode=self.mode, **self.pad_kwargs)
            
        if self.random_center:
            valid_size = get_valid_patch_size(img_size, self._size)
            self._slices = get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img: torch.Tensor, randomize: bool = True) -> torch.Tensor:  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.

        """
        if randomize:
            self.randomize(img.shape[1:])
        if self._size is None:
            raise RuntimeError("self._size not specified.")
        img = self._padder(img)
        if self.random_center:
            return super().__call__(img=img, slices=self._slices)
        cropper = CenterSpatialCrop(self._size)
        return super().__call__(img=img, slices=cropper.compute_slices(img.shape[1:]))
    
class RandSpatialPadCropd(RandCropd):
    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        max_roi_size: Optional[Union[Sequence[int], int]] = None,
        random_center: bool = True,
        pad_tolerance: float = 1,
        method: str = Method.SYMMETRIC,
        mode: str = PytorchPadMode.CONSTANT,
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ) -> None:
        cropper = RandSpatialPadCrop(roi_size, max_roi_size=max_roi_size, random_center=random_center, pad_tolerance=pad_tolerance, method=method, mode=mode, **pad_kwargs)
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys)
        
    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        # the first key must exist to execute random operations
        self.randomize(d[self.first_key(d)].shape[1:])
        for key in self.key_iterator(d):
            kwargs = {"randomize": False} if isinstance(self.cropper, Randomizable) else {}
            d[key] = self.cropper(d[key], **kwargs)  # type: ignore
        return d
    
class PickChannel(Transform):
    def __init__(self, channels: Union[List[int], int, None] = None) -> None:
        self.channels = channels
        
    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        if self.channels is None:
            return img
        elif isinstance(self.channels, int):
            return img[:self.channels]
        else:
            return img[self.channels]

class PickChanneld(MapTransform):
    def __init__(self, keys: KeysCollection, *args, **kwargs) -> None:
        super().__init__(keys)
        self.transform = PickChannel(*args, **kwargs)
        
    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Mapping[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.transform.__call__(d[key])
        return d
    

'''
CropForeground with allowing margin as float
'''
def generate_spatial_bounding_box_v2(
    img: NdarrayOrTensor,
    select_fn: Callable = is_positive,
    channel_indices: Optional[IndexSelection] = None,
    margin: Union[Sequence[int], int, Sequence[float], float] = 0,
    allow_smaller: bool = True,
) -> Tuple[List[int], List[int]]:
    
    spatial_size = img.shape[1:]
    data = img[list(ensure_tuple(channel_indices))] if channel_indices is not None else img
    data = select_fn(data).any(0)
    ndim = len(data.shape)
    margin = ensure_tuple_rep(margin, ndim)
    if isinstance(margin[0], float):
        margin = tuple(int(i*j) for i,j in zip(margin, data.shape))
    
    for m in margin:
        if m < 0:
            raise ValueError("margin value should not be negative number.")

    box_start = [0] * ndim
    box_end = [0] * ndim

    for di, ax in enumerate(itertools.combinations(reversed(range(ndim)), ndim - 1)):
        dt = data
        if len(ax) != 0:
            dt = any_np_pt(dt, ax)

        if not dt.any():
            # if no foreground, return all zero bounding box coords
            return [0] * ndim, [0] * ndim

        arg_max = where(dt == dt.max())[0]
        min_d = arg_max[0] - margin[di]
        max_d = arg_max[-1] + margin[di] + 1
        if allow_smaller:
            min_d = max(min_d, 0)
            max_d = min(max_d, spatial_size[di])

        box_start[di] = min_d.detach().cpu().item() if isinstance(min_d, torch.Tensor) else min_d
        box_end[di] = max_d.detach().cpu().item() if isinstance(max_d, torch.Tensor) else max_d

    return box_start, box_end

class CropForegroundV2(Crop):
    def __init__(
        self,
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int, Sequence[float], float] = 0,
        allow_smaller: bool = True,
        return_coords: bool = False,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ) -> None:
        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.allow_smaller = allow_smaller
        self.return_coords = return_coords
        self.k_divisible = k_divisible
        self.padder = Pad(mode=mode, **pad_kwargs)

    def compute_bounding_box(self, img: torch.Tensor):
        box_start, box_end = generate_spatial_bounding_box_v2(
            img, self.select_fn, self.channel_indices, self.margin, self.allow_smaller
        )
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size
        return box_start_, box_end_

    def crop_pad(
        self, img: torch.Tensor, box_start: np.ndarray, box_end: np.ndarray, mode: Optional[str] = None, **pad_kwargs
    ):
        """
        Crop and pad based on the bounding box.
        """
        slices = self.compute_slices(roi_start=box_start, roi_end=box_end)
        cropped = super().__call__(img=img, slices=slices)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(box_end - np.asarray(img.shape[1:]), 0)
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        pad_width = BorderPad(spatial_border=pad).compute_pad_width(cropped.shape[1:])
        ret = self.padder.__call__(img=cropped, to_pad=pad_width, mode=mode, **pad_kwargs)
        # combine the traced cropping and padding into one transformation
        # by taking the padded info and placing it in a key inside the crop info.
        if get_track_meta():
            ret_: MetaTensor = ret  # type: ignore
            app_op = ret_.applied_operations.pop(-1)
            ret_.applied_operations[-1][TraceKeys.EXTRA_INFO]["pad_info"] = app_op
        return ret

    def __call__(self, img: torch.Tensor, mode: Optional[str] = None, **pad_kwargs):  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = self.compute_bounding_box(img)
        cropped = self.crop_pad(img, box_start, box_end, mode, **pad_kwargs)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.get_most_recent_transform(img)
        # we moved the padding info in the forward, so put it back for the inverse
        pad_info = transform[TraceKeys.EXTRA_INFO].pop("pad_info")
        img.applied_operations.append(pad_info)
        # first inverse the padder
        inv = self.padder.inverse(img)
        # and then inverse the cropper (self)
        return super().inverse(inv)
    
class CropForegroundV2d(Cropd):

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int, Sequence[float], float] = 0,
        allow_smaller: bool = True,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: SequenceStr = PytorchPadMode.CONSTANT,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ) -> None:

        self.source_key = source_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        cropper = CropForegroundV2(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=margin,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            **pad_kwargs,
        )
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.cropper: CropForegroundV2
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key])
        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)
        return d