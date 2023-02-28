


from __future__ import annotations

from typing import Optional, Union
import warnings

import numpy as np
import torch

from monai.metrics.utils import (
    do_metric_reduction,
    get_mask_edges,
    get_surface_distance,
    ignore_background,
    is_binary_tensor,
)
from monai.utils import MetricReduction, convert_data_type

from monai.metrics import HausdorffDistanceMetric

class MedPyHausdorffDistanceMetric(HausdorffDistanceMetric):
    """
    MONAI's percentile Hausdorff Distance is slightly different from MedPy
    This version is same as medpy.metric.binary.hd95
    """
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.
        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")

        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        # compute (BxC) for each channel for each batch
        return compute_hausdorff_distance_medpy(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
        )

def compute_hausdorff_distance_medpy(
    y_pred: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: float | None = None,
    directed: bool = False,
) -> torch.Tensor:
    """
    Compute the Hausdorff distance.
    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    batch_size, n_class = y_pred.shape[:2]
    hd = np.empty((batch_size, n_class))
    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt) = get_mask_edges(y_pred[b, c], y[b, c])
        if not np.any(edges_gt):
            warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan/inf distance.")
        if not np.any(edges_pred):
            warnings.warn(f"the prediction of class {c} is all 0, this may result in nan/inf distance.")

        surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric=distance_metric)
        if not directed:
            distance_2 = get_surface_distance(edges_gt, edges_pred, distance_metric=distance_metric)
            surface_distance = np.hstack((surface_distance, distance_2))

        if surface_distance.shape == (0,):
            bc_hd = np.nan
        if not percentile:
            bc_hd = surface_distance.max()  # type: ignore[no-any-return]
        elif 0 <= percentile <= 100:
            bc_hd = np.percentile(surface_distance, percentile)  # type: ignore[no-any-return]
        else:
            raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")
        hd[b, c] = bc_hd

    return convert_data_type(hd, output_type=torch.Tensor, device=y_pred.device, dtype=torch.float)[0]
