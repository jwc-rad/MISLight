import functools
import numpy as np
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
import torch
import torch.nn.functional as F

########################################################
# torch.nn.functional.interpolate-based resample
# only supports nearest and trilinear interpolation for 3D images
########################################################

class TorchResample(object):
    '''
    Resample image expects image shape (D,W,H) or (C,D,W,H)
    Resample mask expects mask shape (D,W,H) [int label], (1,D,W,H) [int label], or (C,D,W,H) [softmax float]
    '''
    def __init__(self, spatial_dims=3, mode='area', align_corners=None, antialias=False):
        self.spatial_dims = spatial_dims
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias
                
    def resample_to_size(self, npy_image, target_size, device='cpu', mode=None, align_corners=None, antialias=None):
        mode = self.mode if mode is None else mode
        align_corners = self.align_corners if align_corners is None else align_corners
        antialias = self.antialias if antialias is None else antialias
        
        #source_size = npy_image.shape[-self.spatial_dims:]
        #zoom_factor = source_size / np.array(target_size)
        
        original_shape = npy_image.shape
        if len(npy_image.shape)==self.spatial_dims:
            npy_image = np.expand_dims(npy_image, axis=0)
        npy_image = np.expand_dims(npy_image, axis=0)
        
        #if any(np.issubdtype(npy_image.dtype, x) for x in [np.integer, bool]):
        #    npy_image = npy_image.astype(np.int16)
        
        target_tensor = torch.from_numpy(npy_image).to(torch.device(device if torch.cuda.is_available() else 'cpu'))
        # trilinear, nearest upsample is not supported for half precision
        #if any(np.issubdtype(npy_image.dtype, x) for x in [np.integer, bool, np.float16]) and (mode in ['trilinear', 'nearest']):
        #    target_tensor = target_tensor.type(torch.float32)
        target_tensor = target_tensor.float()

        target_tensor = F.interpolate(target_tensor, size=tuple(target_size), mode=mode, align_corners=align_corners, antialias=antialias)
        target_npy_image = target_tensor.cpu().numpy().astype(npy_image.dtype)[0]
        
        if len(original_shape)==self.spatial_dims:
            target_npy_image = target_npy_image[0]
        
        return target_npy_image
    
    def resample_to_spacing(self, npy_image, source_spacing, target_spacing, device='cpu', mode=None, align_corners=None, antialias=None):
        zoom_factor = np.array(target_spacing) / np.array(source_spacing)
        target_size = npy_image.shape / zoom_factor
        target_size = np.rint(target_size).astype(int)
        
        target_npy_image = self.resample_to_size(npy_image, target_size, device, mode=mode, align_corners=align_corners, antialias=antialias)
        
        return target_npy_image
        
    def resample_mask_to_size(self, npy_mask, target_size, max_label, device='cpu', argmax=False, mode=None, align_corners=None, antialias=None):        
        mode = self.mode if mode is None else mode
        align_corners = self.align_corners if align_corners is None else align_corners
        antialias = self.antialias if antialias is None else antialias
        
        #source_size = npy_mask.shape[-self.spatial_dims:]
        #zoom_factor = source_size / np.array(target_size)
        
        if any(np.issubdtype(npy_mask.dtype, x) for x in [np.integer, bool]):
            mask_dtype = np.float16
        else:
            mask_dtype = npy_mask.dtype
        
        original_shape = npy_mask.shape
        # to (1,C,D,W,H)
        if len(npy_mask.shape)==self.spatial_dims:
            npy_mask = np.expand_dims(npy_mask, axis=0)
        npy_mask = np.expand_dims(npy_mask, axis=0)
        
        # change int label to one hot
        if npy_mask.shape[1] == 1:
            label_set = [i for i in range(0, max_label + 1) if i in npy_mask]            
            temp_masks = []
            for i in label_set:
                temp_masks.append((npy_mask==i).astype(mask_dtype))
            npy_mask = np.concatenate(temp_masks, axis=1)
        
        target_tensor = torch.from_numpy(npy_mask).to(torch.device(device if torch.cuda.is_available() else 'cpu'))
        ## trilinear, nearest upsample is not supported for half precision
        #if (target_tensor.dtype == torch.float16) and (mode in ['trilinear', 'nearest']):
        #    target_tensor = target_tensor.type(torch.float32)
        target_tensor = target_tensor.float()
        
        target_tensor = F.interpolate(target_tensor, size=tuple(target_size), mode=mode, align_corners=align_corners, antialias=antialias)[0]
        
        # back to input shape
        if len(original_shape)==self.spatial_dims:
            _target = target_tensor.argmax(0).cpu().numpy()
            target_npy_mask = np.zeros_like(_target)
            for i,x in enumerate(label_set):
                target_npy_mask[_target==i] = x            
        else:
            if original_shape[0]==1:
                _target = target_tensor.argmax(0, True).cpu().numpy()
                target_npy_mask = np.zeros_like(_target)
                for i,x in enumerate(label_set):
                    target_npy_mask[_target==i] = x
            else:
                if argmax:
                    target_npy_mask = target_tensor.argmax(0).cpu().numpy()
                else:
                    target_npy_mask = target_tensor.cpu().numpy().astype(mask_dtype)
            
        return target_npy_mask

    def resample_mask_to_spacing(self, npy_mask, source_spacing, target_spacing, max_label, device='cpu', argmax=False, mode=None, align_corners=None, antialias=None):  
        zoom_factor = np.array(target_spacing) / np.array(source_spacing)
        target_size = npy_mask.shape[-3:] / zoom_factor
        target_size = np.rint(target_size).astype(int)

        target_npy_mask = self.resample_mask_to_size(npy_mask, target_size, max_label, device, argmax, mode=mode, align_corners=align_corners, antialias=antialias)
                    
        return target_npy_mask