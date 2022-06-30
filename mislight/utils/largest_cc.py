'''Adapted from https://github.com/Shanghai-Aitrox-Technology/EfficientSegmentation/blob/master/Common/mask_process.py
'''

import cc3d
from collections import OrderedDict
import fastremap
import numpy as np
from typing import List

def extract_topk_largest_candidates(npy_mask: np.array, label_topk: List, area_least: int = 0) -> np.array:
    '''npy_mask = (H,W,D) int
    '''
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = OrderedDict(sorted(areas.items(), key=lambda item: item[1], reverse=True))
    
    out_mask = np.zeros(npy_mask.shape, npy_mask.dtype)
    for i, topk in enumerate(label_topk):
        out_label = i+1
        label_cc_keys = fastremap.unique(labels_out[npy_mask==out_label])
        cnt = 0
        for k in candidates.keys():
            if k in label_cc_keys:
                out_mask[labels_out == int(k)] = out_label
                cnt += 1
            if cnt >= topk:
                break
    return 
