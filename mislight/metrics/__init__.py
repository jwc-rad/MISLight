import argparse
from collections import OrderedDict
import itertools
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl

from monai.metrics import MeanIoU, ConfusionMatrixMetric

def define_metrics(opt_metric):
    metrics = {}
    if opt_metric:
        mets = opt_metric.lower().split('_')
        if 'iou' in mets:
            #metrics['mIoU'] = MeanIoU(include_background=False) # per-image iou
            metrics['mIoU'] = ConfusionMatrixMetric(include_background=False, metric_name='threat score') # aggregate all confusion matrix and then calculate IoU
        if 'f1' in mets:
            metrics['F1'] = ConfusionMatrixMetric(include_background=False, metric_name='f1 score')
            
        # MULTI LABEL
        if 'mliou' in mets:
            metrics['mIoU'] = ConfusionMatrixMetric(include_background=True, metric_name='threat score')
        if 'mlf1' in mets:
            metrics['F1'] = ConfusionMatrixMetric(include_background=True, metric_name='f1 score')
        
    return metrics