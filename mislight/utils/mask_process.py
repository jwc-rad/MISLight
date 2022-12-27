import argparse
import glob
from imageio import imread, imwrite
import numpy as np
import os
from PIL import Image
import shutil
from tqdm.autonotebook import tqdm

def convert_array_at_once(x, convert_dict):
    select_k = []
    select_v = []
    for k,v in convert_dict.items():
        select_k.append(x==k)
        select_v.append(v)
    return np.select(select_k, select_v, x)

def convert_tooth_to_mask(x):
    convert_dict= {
        11: 1,
        12: 2,
        13: 3,
        14: 4,
        15: 5,
        16: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        26: 12,
        31: 13,
        32: 14,
        33: 15,
        34: 16,
        35: 17,
        36: 18,
        41: 19,
        42: 20,
        43: 21,
        44: 22,
        45: 23,
        46: 24,
    }
    return convert_array_at_once(x, convert_dict)

def convert_mask_to_tooth(x):
    convert_dict = {
        1: 11,
        2: 12,
        3: 13,
        4: 14,
        5: 15,
        6: 16,
        7: 21,
        8: 22,
        9: 23,
        10: 24,
        11: 25,
        12: 26,
        13: 31,
        14: 32,
        15: 33,
        16: 34,
        17: 35,
        18: 36,
        19: 41,
        20: 42,
        21: 43,
        22: 44,
        23: 45,
        24: 46
    }
    return convert_array_at_once(x, convert_dict)

def postprocess_view(x, view):
    if view == 'front':
        convert_dict= {
            33: 0,
            34: 0,
            11: 0,
            12: 0,
            17: 0,
            18: 0,
            23: 0,
            24: 0,
        }        
        return convert_array_at_once(x, convert_dict)
    elif view == 'left':
        convert_dict= {
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
        }        
        return convert_array_at_once(x, convert_dict)
    elif view == 'right':
        convert_dict= {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            19: 0,
            20: 0,
            21: 0,
            22: 0,
            23: 0,
            24: 0,
        }        
        return convert_array_at_once(x, convert_dict)
    elif view == 'upper':
        convert_dict= {
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            19: 0,
            20: 0,
            21: 0,
            22: 0,
            23: 0,
            24: 0,
        }        
        return convert_array_at_once(x, convert_dict)
    elif view == 'lower':
        convert_dict= {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0,
            10: 0,
            11: 0,
            12: 0,
        }        
        return convert_array_at_once(x, convert_dict)
    else:
        raise NotImplementedError(f'view [{view}] is not recognized') 
    return None
    
    