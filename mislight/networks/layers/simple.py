from copy import deepcopy
import functools
import math
import numpy as np
from typing import Optional, Sequence, Tuple, Union
import torch
from torch import nn

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
        
