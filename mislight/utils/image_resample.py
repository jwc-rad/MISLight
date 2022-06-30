import functools
import numpy as np
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
import torch
import torch.nn.functional as F

########################################################
# helper functions
########################################################

def order2mode(order, dim=3):
    mode = None
    if order == 0:
        mode = 'nearest'
    elif order == 1:
        if dim == 3:
            mode = 'trilinear'
        elif dim == 2:
            mode = 'bilinear'
    elif order == 3:
        if dim == 2:
            mode = 'bicubic'
    if mode is None:
        raise NotImplementedError(f'interpolation order [{order}] is not implemented!')
    return mode

########################################################
# torch.nn.functional.interpolate-based resample
# only supports nearest and trilinear interpolation for 3D images
########################################################

class TorchResample(object):
    '''
    Resample image expects image shape (D,W,H) or (C,D,W,H)
    Resample mask expects mask shape (D,W,H) [int label], (1,D,W,H) [int label], or (C,D,W,H) [softmax float]
    '''
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def resample_to_size(npy_image, target_size, order=1, device='cpu'):
        mode = order2mode(order)
            
        source_size = npy_image.shape
        zoom_factor = source_size / np.array(target_size)
        
        original_shape = npy_image.shape
        if len(npy_image.shape)==3:
            npy_image = np.expand_dims(npy_image, axis=0)
        npy_image = np.expand_dims(npy_image, axis=0)
        
        if any(np.issubdtype(npy_image.dtype, x) for x in [np.integer, bool]):
            npy_image = npy_image.astype(np.int16)
        
        target_tensor = torch.from_numpy(npy_image).to(torch.device(device if torch.cuda.is_available() else 'cpu'))
        # trilinear, nearest upsample is not supported for half precision
        if any(np.issubdtype(npy_image.dtype, x) for x in [np.integer, bool, np.float16]) and (mode in ['trilinear', 'nearest']):
            target_tensor = target_tensor.type(torch.float32)
        target_tensor = F.interpolate(target_tensor, size=tuple(target_size), mode=mode, align_corners=True if order>0 else None)
        target_npy_image = target_tensor.cpu().numpy().astype(npy_image.dtype)[0]
        
        if len(original_shape)==3:
            target_npy_image = target_npy_image[0]
        
        return target_npy_image, zoom_factor
    
    @staticmethod
    def resample_to_spacing(npy_image, source_spacing, target_spacing, order=1, device='cpu'):
        zoom_factor = np.array(target_spacing) / np.array(source_spacing)
        target_size = npy_image.shape / zoom_factor
        target_size = np.rint(target_size).astype(int)
        
        target_npy_image, zoom_factor = TorchResample.resample_to_size(npy_image, target_size, order, device)
        
        return target_npy_image, zoom_factor    
        
    @staticmethod
    def resample_mask_to_size(npy_mask, target_size, num_label, order=1, device='cpu', argmax=False):
        mode = order2mode(order)
        
        source_size = npy_mask.shape[-3:]
        zoom_factor = source_size / np.array(target_size)
        
        if any(np.issubdtype(npy_mask.dtype, x) for x in [np.integer, bool]):
            mask_dtype = np.float16
        else:
            mask_dtype = npy_mask.dtype
        
        original_shape = npy_mask.shape
        # to (1,C,D,W,H)
        if len(npy_mask.shape)==3:
            npy_mask = np.expand_dims(npy_mask, axis=0)
        npy_mask = np.expand_dims(npy_mask, axis=0)
        
        # change int label to one hot
        if npy_mask.shape[1] == 1:
            temp_masks = []
            for i in range(0, num_label + 1):
                temp_masks.append((npy_mask==i).astype(mask_dtype))
            npy_mask = np.concatenate(temp_masks, axis=1)
        
        target_tensor = torch.from_numpy(npy_mask).to(torch.device(device if torch.cuda.is_available() else 'cpu'))
        # trilinear, nearest upsample is not supported for half precision
        if (target_tensor.dtype == torch.float16) and (mode in ['trilinear', 'nearest']):
            target_tensor = target_tensor.type(torch.float32)
        target_tensor = F.interpolate(target_tensor, size=tuple(target_size), mode=mode, align_corners=True if order>0 else None)[0]
        
        # back to input shape
        if len(original_shape)==3:
            target_npy_mask = target_tensor.argmax(0).cpu().numpy()
        else:
            if original_shape[0]==1:
                target_npy_mask = target_tensor.argmax(0, True).cpu().numpy()
            else:
                if argmax:
                    target_npy_mask = target_tensor.argmax(0).cpu().numpy()
                else:
                    target_npy_mask = target_tensor.cpu().numpy().astype(mask_dtype)
            
        return target_npy_mask, zoom_factor

    @staticmethod
    def resample_mask_to_spacing(npy_mask, source_spacing, target_spacing, num_label, order=1, device='cpu', argmax=False):  
        zoom_factor = np.array(target_spacing) / np.array(source_spacing)
        target_size = npy_mask.shape[-3:] / zoom_factor
        target_size = np.rint(target_size).astype(int)

        target_npy_mask, zoom_factor = TorchResample.resample_mask_to_size(npy_mask, target_size, num_label, order, device, argmax)
                    
        return target_npy_mask, zoom_factor
    
