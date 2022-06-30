import numbers
import numpy as np
import random

import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms

from mislight.utils.image_resample import order2mode
from mislight.utils.spatial_transformations import matrix_rotation3d_x, matrix_rotation3d_y, matrix_rotation3d_z, matrix_scale3d, matrix_identity3d, generate_deformation_grid

def get_transforms(phase, opt):
    transform_list = []
    transform_list += [ToTensor3D()]
    if phase == 'test':
        transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
        transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []    
        transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]   
    elif phase == 'valid':
        transform_list += [AddHU(1024)]
        transform_list += [Padding3D(opt.crop_size)]
        transform_list += [CenterCrop3D(opt.crop_size)]  
        transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
        transform_list += [AddHU(-1024)]
        transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []    
        transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
    else:
        if opt.data_augmentation == 'v0':
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(opt.crop_size)]
            transform_list += [RandomCrop3D(opt.crop_size)]
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [AddHU(-1024)]
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
        elif opt.data_augmentation == 'v4':
            buffer_crop = np.rint(np.array(opt.crop_size) * np.array([1.25])).astype(int)
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(buffer_crop)]
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [RandomElastic3D(sigma_factor=(0.75, 1), p=0.5, num_points=16)]
            transform_list += [RandomAffine3D(scale=(0.5,2,0.8,1.25,0.8,1.25), rot=(-15,15))]
            transform_list += [AddHU(-1024)]
            #transform_list += [CenterCrop3D(opt.crop_size)] 
            transform_list += [RandomCrop3D(opt.crop_size)]
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
            transform_list += [NoiseGaussian()]
            transform_list += [AugmentBrightnessMultiplicative()]
            transform_list += [BackgroundLabelProcess(opt.num_classes+1, 1)]
        else:
            raise NotImplementedError(f'data augmentation [{opt.data_augmentation}] is not implemented!')       
    return transform_list
    
'''
Numpy array shape is expected to be (channel,z,y,x) like pytorch
Label array/tensor is (1,z,y,x) after CreateOnehotLabel
'''
    
############################################################
## Random Erasing (https://arxiv.org/abs/1708.04896)
## adapted from torchvision.transforms.RandomErasing
############################################################
    
class RandomErasing3D(object):
    '''
    Randomly selects a cube region in an torch Tensor image and erases its pixels and change labels to background
    Args:
        scale: range of proportion of erased area against input image
        ratio: range of aspect ratio of erased area.
        value: erasing value. If none, uniform random between min~max
        ** torchvision 2D defaults: scale=(0.02, 0.33), ratio=(0.3, 3.3)
    '''
    def __init__(self, p=0.5, scale=(0.003, 0.2), ratio=(0.3, 3.3), value=None):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value 

    @staticmethod
    def get_params(img, scale, ratio, value=None):
        img_c, img_d, img_h, img_w = img.shape[-4], img.shape[-3], img.shape[-2], img.shape[-1]
        volume = img_d * img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_volume = volume * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio1 = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            aspect_ratio2 = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            d = int(round((erase_volume * aspect_ratio1)) ** (1/3.))
            h = int(round((erase_volume * aspect_ratio2)) ** (1/3.))
            w = int(round((erase_volume / aspect_ratio1 / aspect_ratio2)) ** (1/3.))
            if not (d < img_d and h < img_h and w < img_w):
                continue

            if value is None:
                #M = img.max()
                #m = img.min()
                #cval = torch.empty(1).uniform_(m, M).item()
                #v = torch.full([img_c, d, h, w], cval, device=img.device)
                v = torch.empty([img_c, d, h, w], dtype=torch.float32, device=img.device).normal_()
            else:
                v = torch.tensor(value, device=img.device)[:, None, None, None]

            i = torch.randint(0, img_d - d + 1, size=(1,)).item()
            j = torch.randint(0, img_h - h + 1, size=(1,)).item()
            k = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, k, d, h, w, v

        # Return original image
        return 0, 0, 0, img_d, img_h, img_w, img
        
    def __call__(self, sample):
        x = sample['image']
        if 'label' in sample.keys():
            y = sample['label']
        
        if np.random.uniform() < self.p:
            i, j, k, d, h, w, v = self.get_params(x, scale=self.scale, ratio=self.ratio, value=self.value)
            x[:, i:i+d, j:j+h, k:k+w] = v
            y[:, i:i+d, j:j+h, k:k+w] = 0
            y[0, i:i+d, j:j+h, k:k+w] = 1
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            return_items['label'] = y

        return return_items
    
############################################################
## Image Augmentations
## adapted from https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations
############################################################  
    
class RandomElastic3D(object):
    '''
    Elastic Deformation
    Only for Tensor
    Apply if np.random.uniform() < p
    '''
    def __init__(self, sigma_factor=(0.75, 1), p=0.5, num_points=16, truncate=4, align_corners=True, order_image=1, order_label=1):
        
        self.sigma_factor = sigma_factor
        self.p = p
        self.num_points = num_points
        self.truncate = truncate
        self.align_corners = align_corners
        self.order_image = order_image
        self.order_label = order_label
        
    def __call__(self, sample):
        x = sample['image']
        
        modified = False
        if np.random.uniform() < self.p:
            modified = True
            x = x.unsqueeze(0)
            deformation = generate_deformation_grid(x.shape, self.num_points, self.sigma_factor, self.truncate, self.order_image, self.align_corners, x.device)
            x = F.grid_sample(x, deformation, mode=order2mode(self.order_image, dim=2), align_corners=self.align_corners).squeeze(0)
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            if modified:
                y = F.grid_sample(y.unsqueeze(0), deformation, mode=order2mode(self.order_label, dim=2), align_corners=self.align_corners).squeeze(0)
            return_items['label'] = y
        return return_items 
        
class RandomAffine3D(object):
    '''
    Scale and Rotation
    Only for Tensor
    Apply if np.random.uniform() < p
    '''
    def __init__(self, scale=(0.5,2,0.8,1.25,0.8,1.25), p_scale=0.5, rot=(-15,15), p_rot=0.5, align_corners=True):
        scale = scale * 3
        self.scale_z = scale[:2]
        self.scale_y = scale[2:4]
        self.scale_x = scale[4:6]
        self.rot = rot
        self.p_scale = p_scale
        self.p_rot = p_rot
        self.align_corners = align_corners
        
    def __call__(self, sample):
        x = sample['image']
        
        affines = []
        
        sz, sy, sx = 1,1,1
        if np.random.uniform() < self.p_scale:
            sz = np.random.uniform(*self.scale_z)
        if np.random.uniform() < self.p_scale:
            sy = np.random.uniform(*self.scale_y)
        if np.random.uniform() < self.p_scale:
            sx = np.random.uniform(*self.scale_x)
        affines.append(matrix_scale3d((sz,sy,sx)))
        
        if np.random.uniform() < self.p_rot:
            degz = np.random.uniform(*self.rot)
            affines.append(matrix_rotation3d_z(degz))
        if np.random.uniform() < self.p_rot:
            degy = np.random.uniform(*self.rot)
            affines.append(matrix_rotation3d_y(degy))
        if np.random.uniform() < self.p_rot:
            degx = np.random.uniform(*self.rot)
            affines.append(matrix_rotation3d_x(degx))

        affine_matrix = matrix_identity3d()
        for aff in affines:
            affine_matrix = np.dot(aff, affine_matrix)
    
        affgrid = F.affine_grid(torch.from_numpy(affine_matrix[:-1]).type(x.dtype).unsqueeze(0), x.unsqueeze(0).shape, align_corners=self.align_corners)
        affgrid = affgrid.to(x.device)
        x = F.grid_sample(x.unsqueeze(0), affgrid, align_corners=self.align_corners).squeeze(0)
        del affgrid
        
        return_items = {'image': x}

        if 'label' in sample.keys():
            y = sample['label']
            affgrid = F.affine_grid(torch.from_numpy(affine_matrix[:-1]).type(y.dtype).unsqueeze(0), y.unsqueeze(0).shape, align_corners=self.align_corners)
            affgrid = affgrid.to(x.device)
            y = F.grid_sample(y.unsqueeze(0), affgrid, align_corners=self.align_corners).squeeze(0)
            del affgrid
            return_items['label'] = y
        return return_items 
        
class RandomCrop3D(object):
    '''
    Random crop
    Args:
        crop_size: ZYX
    '''

    def __init__(self, crop_size):
        self.z_crop, self.y_crop, self.x_crop = crop_size
        
    def __call__(self, sample):
        x = sample['image']
        
        iz, iy, ix = x.shape[-3:]    
        zmin = random.randint(0, np.maximum(0, iz - self.z_crop))
        ymin = random.randint(0, np.maximum(0, iy - self.y_crop))
        xmin = random.randint(0, np.maximum(0, ix - self.x_crop))
        zmax = zmin + self.z_crop
        ymax = ymin + self.y_crop
        xmax = xmin + self.x_crop
            
        slicer = ((slice(zmin,zmax),slice(ymin,ymax),slice(xmin,xmax)))
        return_items = {'image': x[(slice(None),)*max(0,len(x.shape)-3)+slicer]}
                    
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y[(slice(None),)*max(0,len(y.shape)-3)+slicer]
        return return_items
    
class RandomMirror3D(object):
    '''
    Mirror x,y,z
    Args:
        axes: 0,1,2 = z,y,x
        p: apply if np.random.uniform() < p * 0.5
    '''
    def __init__(self, axes=(0,1,2), p=1):
        self.axes = axes
        self.p = p
        
    def __call__(self, sample):
        x = sample['image']
            
        slices = [0,0,0]
        for i in self.axes:
            if np.random.uniform() < self.p * 0.5:
                slices[i] = 1
        slices = [0] + slices
        flip_axes = tuple(np.argwhere(slices)[:,0])
        
        if len(flip_axes) > 0:
            if isinstance(x, Tensor):
                return_items = {'image': torch.flip(x, flip_axes)}   
            else:
                return_items = {'image': np.flip(x, flip_axes)}
        else:
            return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            if len(flip_axes) > 0:
                if isinstance(y, Tensor):
                    return_items = {'label': torch.flip(y, flip_axes)}   
                else:
                    return_items = {'label': np.flip(y, flip_axes)}
            else:
                return_items = {'label': y}            
        return return_items
    
class NoiseGaussian(object):
    '''
    Gaussian noise
    Args:
        noise_variance: range of noise variance in np.random.normal(0, variance)
        p: apply if np.random.uniform() < p
    '''    
    def __init__(self, noise_variance=(0,0.1), p=1):
        self.p = p
        self.noise_variance = noise_variance
        
    def __call__(self, sample):
        x = sample['image']
        
        for c in range(x.shape[0]):
            if np.random.uniform() < self.p:
                varmin, varmax = self.noise_variance
                if varmin == varmax:
                    variance = varmin
                else:
                    variance = random.uniform(varmin, varmax)
                if isinstance(x, Tensor):
                    x[c] += torch.normal(0.0, variance, size=x[c].shape, device=x.device)
                else:
                    x[c] += np.random.normal(0.0, variance, size=x[c].shape)
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items   
        
class AugmentBrightnessMultiplicative(object):
    '''
    Multiplicative augmentation on brightness
    Args:
        factor_range
        p: apply if np.random.uniform() < p
    '''    
    def __init__(self, factor_range=(0.75, 1.25), p=0.5):
        self.p = p
        self.factor_range = factor_range
        
    def __call__(self, sample):
        x = sample['image']
        
        for c in range(x.shape[0]):
            if np.random.uniform() < self.p:
                factor = np.random.uniform(*self.factor_range)
                x[c] *= factor
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items 
        
# almost same as multiplicative brightness for normalized input
class AugmentContrast(object):
    '''
    Augmentation on contrast
    Args:
        factor_range
        p: apply if np.random.uniform() < p
    '''    
    def __init__(self, factor_range=(0.75, 1.25), p=0.15, preserve_range=True):
        self.p = p
        self.factor_range = factor_range
        self.preserve_range = preserve_range
        
    def __call__(self, sample):
        x = sample['image']
        
        for c in range(x.shape[0]):
            if np.random.uniform() < self.p:                
                if np.random.random() < 0.5 and self.factor_range[0] < 1:
                    factor = np.random.uniform(self.factor_range[0], 1)
                else:
                    factor = np.random.uniform(max(self.factor_range[0], 1), self.factor_range[1])

                mn = x[c].mean()
                if self.preserve_range:
                    minm = x[c].min()
                    maxm = x[c].max()

                x[c] = (x[c] - mn) * factor + mn

                if self.preserve_range:
                    x[c][x[c] < minm] = minm
                    x[c][x[c] > maxm] = maxm
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items 


############################################################
## Basic Transforms
############################################################    

class ClipHU(object):
    def __init__(self, window_width, window_level):
        self.clip_min = window_level - 0.5*window_width
        self.clip_max = window_level + 0.5*window_width
        
    def __call__(self, sample):
        x = sample['image']
    
        x[x<self.clip_min] = self.clip_min
        x[x>self.clip_max] = self.clip_max
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items
    
class AddHU(object):
    '''torch grid_sample zero-padding should be -1024HU
    '''
    def __init__(self, value=1024):
        self.value = value
        
    def __call__(self, sample):
        x = sample['image']
        x = x + self.value
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items
    
class Normalize(object):
    def __init__(self, global_mean=None, global_std=None):
        self.global_mean = global_mean
        self.global_std = global_std
        
    def __call__(self, sample):
        x = sample['image']
        
        if not self.global_mean or not self.global_std:
            dims = tuple(np.arange(len(x.shape))[1:])
            if isinstance(x, Tensor):
                std, mean = torch.std_mean(x, dim=dims, keepdim=True)
            else:
                mean = np.mean(x, axis=dims, keepdims=True)
                std = np.std(x, axis=dims, keepdims=True)
        else:
            mean, std = self.global_mean, self.global_std
        
        x = (x-mean)/(std + 1e-5)
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items

class CreateOnehotLabel(object):
    '''
    if onehot, convert gt from shape (b, z, y, x) to one hot encoding (b, z, y, x, c)
    if no onehot, convert gt from shape (b, z, y, x) to shape (b, z, y, x, 1)
    '''
    def __init__(self, num_classes, onehot=True):
        self.num_classes = num_classes
        self.onehot = onehot

    def __call__(self, sample):
        return_items = {'image': sample['image']}
        
        if 'label' in sample.keys():
            y = sample['label']
            if self.onehot:
                if isinstance(y, Tensor):
                    label_dtype = torch.float32
                    onehot_label = torch.zeros((self.num_classes,) + tuple(y.shape), dtype=label_dtype)
                    # dummy label
                    if torch.max(y) >= self.num_classes:
                        onehot_label = onehot_label - 1
                    for i in range(self.num_classes):
                        onehot_label[i] = (y == i).type(label_dtype)
                else:
                    label_dtype = np.float32
                    onehot_label = np.zeros((self.num_classes,) + tuple(y.shape), dtype=label_dtype)
                    # dummy label
                    if y.max() >= self.num_classes:
                        onehot_label = onehot_label - 1
                    for i in range(self.num_classes):
                        onehot_label[i] = (y == i).astype(label_dtype)
            else:
                if isinstance(y, Tensor):
                    onehot_label = torch.unsqueeze(y, 0)
                else:
                    onehot_label = np.expand_dims(y, axis=0)
            
            return_items['label'] = onehot_label

        return return_items
    
class BackgroundLabelProcess(object):
    '''
    For float-type onehot label, change all-zero points to background label and suppress background to not lose foreground labels
    '''
    def __init__(self, num_classes: int, weight=1, eps=1e-7):
        self.num_classes = num_classes
        self.weight = [weight,] + [1,]*(num_classes-1)
        self.eps = eps

    def __call__(self, sample):
        return_items = {'image': sample['image']}
        
        if 'label' in sample.keys():
            y = sample['label']
            y[0] += self.eps
            w = torch.tensor(self.weight, device=y.device)
            wy = y * w[(...,) + (None,)*(len(y.shape)-1)]
            y = wy / wy.sum(0).unsqueeze(0)
            return_items['label'] = y

        return return_items

class ToTensor3D(object):
    def __init__(self):
        pass
    
    def __call__(self, sample):
        x = sample['image']
        X = torch.from_numpy(x).float().contiguous()
        return_items = {'image': X}
        
        if 'label' in sample.keys():
            y = sample['label']
            Y = torch.from_numpy(y).float().contiguous()
            return_items['label'] = Y
        
        return return_items
    
class Padding3D(object):
    '''
    Add padding
    Args:
        size: target size (z,y,x)
        padding_value: constant padding. number or ['min','max','mean']
        center: if True, center padding
    '''

    def __init__(self, size, padding_value=0, center=True):
        self.size = np.array(size)
        assert any(isinstance(padding_value, x) for x in [str, numbers.Number])
        self.pv = padding_value
        self.center = center
        
    def __call__(self, sample):
        x = sample['image']

        input_size = x.shape[-3:]
        pads = self.size - input_size
        pads[pads<0] = 0
        # center padding
        if self.center:
            pads = tuple([(a//2, a-a//2) for a in pads.astype(int)])           
        else:
            pads = tuple([(0, a) for a in pads.astype(int)])   
            
        if isinstance(self.pv, str):
            if self.pv.lower() == 'min':
                pv = x.min()
            elif self.pv.lower() == 'max':
                pv = x.max()
            else:
                pv = x.mean()
        else:
            pv = self.pv
        if isinstance(x, Tensor):
            x = torch.nn.functional.pad(x, tuple(np.array(pads[::-1]).ravel()), mode='constant', value=pv)
        else:
            x = np.pad(x, ((0,0),)*max(0,len(x.shape)-3) + pads, 'constant', constant_values=pv)

        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            # label padding must be background
            if isinstance(x, Tensor):
                y = torch.nn.functional.pad(y, tuple(np.array(pads[::-1]).ravel()), mode='constant', value=0)
            else:
                y = np.pad(y, ((0,0),)*max(0,len(y.shape)-3) + pads, 'constant', constant_values=0)
            return_items['label'] = y

        return return_items
    
class CenterCrop3D(object):
    '''
    Center crop
    Args:
        crop_size: ZYX
    '''

    def __init__(self, crop_size):
        self.z_crop, self.y_crop, self.x_crop = crop_size
        
    def __call__(self, sample):
        x = sample['image']
        
        iz, iy, ix = x.shape[-3:]    
        zmin = iz//2 - self.z_crop//2
        ymin = iy//2 - self.y_crop//2
        xmin = ix//2 - self.x_crop//2
        zmax = zmin + self.z_crop
        ymax = ymin + self.y_crop
        xmax = xmin + self.x_crop
            
        slicer = ((slice(zmin,zmax),slice(ymin,ymax),slice(xmin,xmax)))
        return_items = {'image': x[(slice(None),)*max(0,len(x.shape)-3)+slicer]}
                    
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y[(slice(None),)*max(0,len(y.shape)-3)+slicer]
            
        return return_items
    
class Crop3D(object):
    '''
    Crop
    Args:
        z_min: int, y_min: int, x_min: int, z_max: int, y_max: int, x_max: int
    '''

    def __init__(self, z_min: int, y_min: int, x_min: int, z_max: int, y_max: int, x_max: int):
        assert (z_min < z_max)&(y_min < y_max)&(x_min < x_max)
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        
    def __call__(self, sample):
        x = sample['image']
        slicer = ((slice(self.zmin,self.zmax),slice(self.ymin,self.ymax),slice(self.xmin,self.xmax)))
        return_items = {'image': x[(slice(None),)*max(0,len(x.shape)-3)+slicer]}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y[(slice(None),)*max(0,len(y.shape)-3)+slicer]
            
        return return_itemsimport numbers
import numpy as np
import random

import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms

from mislight.utils.image_resample import order2mode
from mislight.utils.spatial_transformations import matrix_rotation3d_x, matrix_rotation3d_y, matrix_rotation3d_z, matrix_scale3d, matrix_identity3d, generate_deformation_grid

def get_transforms(phase, opt):
    transform_list = []
    transform_list += [ToTensor3D()]
    if phase == 'test':
        transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
        transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []    
        transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]   
    elif phase == 'valid':
        transform_list += [AddHU(1024)]
        transform_list += [Padding3D(opt.crop_size)]
        transform_list += [CenterCrop3D(opt.crop_size)]  
        transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
        transform_list += [AddHU(-1024)]
        transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []    
        transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
    else:
        if opt.data_augmentation == 'v0':
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(opt.crop_size)]
            transform_list += [RandomCrop3D(opt.crop_size)]
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [AddHU(-1024)]
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
        elif opt.data_augmentation == 'v1':
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(opt.crop_size)]
            transform_list += [RandomCrop3D(opt.crop_size)]   
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [AddHU(-1024)]
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
            transform_list += [NoiseGaussian()]
            transform_list += [AugmentBrightnessMultiplicative()]
        elif opt.data_augmentation == 'v2':
            buffer_crop = np.rint(np.array(opt.crop_size) * np.array([1])).astype(int)
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(buffer_crop)]
            transform_list += [RandomCrop3D(buffer_crop)]
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [RandomAffine3D(scale=(0.8,1.25,0.8,1.25,0.8,1.25), rot=(-15,15))]
            transform_list += [AddHU(-1024)]
            transform_list += [CenterCrop3D(opt.crop_size)] 
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
            transform_list += [NoiseGaussian()]
            transform_list += [AugmentBrightnessMultiplicative()]
        elif opt.data_augmentation == 'v3':
            buffer_crop = np.rint(np.array(opt.crop_size) * np.array([1])).astype(int)
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(buffer_crop)]
            transform_list += [RandomCrop3D(buffer_crop)]
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [RandomAffine3D(scale=(0.5,2,0.8,1.25,0.8,1.25), rot=(-15,15))]
            transform_list += [AddHU(-1024)]
            transform_list += [CenterCrop3D(opt.crop_size)] 
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
            transform_list += [NoiseGaussian()]
            transform_list += [AugmentBrightnessMultiplicative()]
        elif opt.data_augmentation == 'v3.1':
            buffer_crop = np.rint(np.array(opt.crop_size) * np.array([1.25])).astype(int)
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(buffer_crop)]
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [RandomAffine3D(scale=(0.5,2,0.8,1.25,0.8,1.25), rot=(-15,15))]
            transform_list += [AddHU(-1024)]
            #transform_list += [CenterCrop3D(opt.crop_size)]
            transform_list += [RandomCrop3D(opt.crop_size)]
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
            transform_list += [NoiseGaussian()]
            transform_list += [AugmentBrightnessMultiplicative()]
            transform_list += [BackgroundLabelProcess(opt.num_classes+1, 1)]
        elif opt.data_augmentation == 'v3.2':
            buffer_crop = np.rint(np.array(opt.crop_size) * np.array([1.25])).astype(int)
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(buffer_crop)]
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [RandomAffine3D(scale=(0.5,2,0.8,1.25,0.8,1.25), rot=(-15,15))]
            transform_list += [AddHU(-1024)]
            #transform_list += [CenterCrop3D(opt.crop_size)]
            transform_list += [RandomCrop3D(opt.crop_size)]
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
            transform_list += [NoiseGaussian()]
            transform_list += [AugmentBrightnessMultiplicative()]
            transform_list += [BackgroundLabelProcess(opt.num_classes+1, 1/3.)]
        elif opt.data_augmentation == 'v4':
            buffer_crop = np.rint(np.array(opt.crop_size) * np.array([1.25])).astype(int)
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(buffer_crop)]
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [RandomElastic3D(sigma_factor=(0.75, 1), p=0.5, num_points=16)]
            transform_list += [RandomAffine3D(scale=(0.5,2,0.8,1.25,0.8,1.25), rot=(-15,15))]
            transform_list += [AddHU(-1024)]
            #transform_list += [CenterCrop3D(opt.crop_size)] 
            transform_list += [RandomCrop3D(opt.crop_size)]
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
            transform_list += [NoiseGaussian()]
            transform_list += [AugmentBrightnessMultiplicative()]
            transform_list += [BackgroundLabelProcess(opt.num_classes+1, 1)]
        elif opt.data_augmentation == 'v5':
            buffer_crop = np.rint(np.array(opt.crop_size) * np.array([1.25])).astype(int)
            transform_list += [AddHU(1024)]
            transform_list += [Padding3D(buffer_crop)]
            transform_list += [CreateOnehotLabel(opt.num_classes+1, True)]
            transform_list += [RandomElastic3D(sigma_factor=(0.75, 1), p=0.5, num_points=16)]
            transform_list += [RandomAffine3D(scale=(0.5,2,0.8,1.25,0.8,1.25), rot=(-15,15))]
            transform_list += [AddHU(-1024)]
            #transform_list += [CenterCrop3D(opt.crop_size)] 
            transform_list += [RandomCrop3D(opt.crop_size)]
            transform_list += [ClipHU(opt.windowHU[0], opt.windowHU[1])] if opt.windowHU else []
            transform_list += [] if opt.no_normalize else [Normalize(opt.global_mean, opt.global_std)]
            transform_list += [RandomErasing3D()]
            transform_list += [NoiseGaussian()]
            transform_list += [AugmentBrightnessMultiplicative()]
            transform_list += [BackgroundLabelProcess(opt.num_classes+1, 1)]
        else:
            raise NotImplementedError(f'data augmentation [{opt.data_augmentation}] is not implemented!')       
    return transform_list
    
'''
Numpy array shape is expected to be (channel,z,y,x) like pytorch
Label array/tensor is (1,z,y,x) after CreateOnehotLabel
'''
    
############################################################
## Random Erasing (https://arxiv.org/abs/1708.04896)
## adapted from torchvision.transforms.RandomErasing
############################################################
    
class RandomErasing3D(object):
    '''
    Randomly selects a cube region in an torch Tensor image and erases its pixels and change labels to background
    Args:
        scale: range of proportion of erased area against input image
        ratio: range of aspect ratio of erased area.
        value: erasing value. If none, uniform random between min~max
        ** torchvision 2D defaults: scale=(0.02, 0.33), ratio=(0.3, 3.3)
    '''
    def __init__(self, p=0.5, scale=(0.003, 0.2), ratio=(0.3, 3.3), value=None):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value 

    @staticmethod
    def get_params(img, scale, ratio, value=None):
        img_c, img_d, img_h, img_w = img.shape[-4], img.shape[-3], img.shape[-2], img.shape[-1]
        volume = img_d * img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_volume = volume * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio1 = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            aspect_ratio2 = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            d = int(round((erase_volume * aspect_ratio1)) ** (1/3.))
            h = int(round((erase_volume * aspect_ratio2)) ** (1/3.))
            w = int(round((erase_volume / aspect_ratio1 / aspect_ratio2)) ** (1/3.))
            if not (d < img_d and h < img_h and w < img_w):
                continue

            if value is None:
                #M = img.max()
                #m = img.min()
                #cval = torch.empty(1).uniform_(m, M).item()
                #v = torch.full([img_c, d, h, w], cval, device=img.device)
                v = torch.empty([img_c, d, h, w], dtype=torch.float32, device=img.device).normal_()
            else:
                v = torch.tensor(value, device=img.device)[:, None, None, None]

            i = torch.randint(0, img_d - d + 1, size=(1,)).item()
            j = torch.randint(0, img_h - h + 1, size=(1,)).item()
            k = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, k, d, h, w, v

        # Return original image
        return 0, 0, 0, img_d, img_h, img_w, img
        
    def __call__(self, sample):
        x = sample['image']
        if 'label' in sample.keys():
            y = sample['label']
        
        if np.random.uniform() < self.p:
            i, j, k, d, h, w, v = self.get_params(x, scale=self.scale, ratio=self.ratio, value=self.value)
            x[:, i:i+d, j:j+h, k:k+w] = v
            y[:, i:i+d, j:j+h, k:k+w] = 0
            y[0, i:i+d, j:j+h, k:k+w] = 1
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            return_items['label'] = y

        return return_items
    
############################################################
## Image Augmentations
## adapted from https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations
############################################################  
    
class RandomElastic3D(object):
    '''
    Elastic Deformation
    Only for Tensor
    Apply if np.random.uniform() < p
    '''
    def __init__(self, sigma_factor=(0.75, 1), p=0.5, num_points=16, truncate=4, align_corners=True, order_image=1, order_label=1):
        
        self.sigma_factor = sigma_factor
        self.p = p
        self.num_points = num_points
        self.truncate = truncate
        self.align_corners = align_corners
        self.order_image = order_image
        self.order_label = order_label
        
    def __call__(self, sample):
        x = sample['image']
        
        modified = False
        if np.random.uniform() < self.p:
            modified = True
            x = x.unsqueeze(0)
            deformation = generate_deformation_grid(x.shape, self.num_points, self.sigma_factor, self.truncate, self.order_image, self.align_corners, x.device)
            x = F.grid_sample(x, deformation, mode=order2mode(self.order_image, dim=2), align_corners=self.align_corners).squeeze(0)
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            if modified:
                y = F.grid_sample(y.unsqueeze(0), deformation, mode=order2mode(self.order_label, dim=2), align_corners=self.align_corners).squeeze(0)
            return_items['label'] = y
        return return_items 
        
class RandomAffine3D(object):
    '''
    Scale and Rotation
    Only for Tensor
    Apply if np.random.uniform() < p
    '''
    def __init__(self, scale=(0.5,2,0.8,1.25,0.8,1.25), p_scale=0.5, rot=(-15,15), p_rot=0.5, align_corners=True):
        scale = scale * 3
        self.scale_z = scale[:2]
        self.scale_y = scale[2:4]
        self.scale_x = scale[4:6]
        self.rot = rot
        self.p_scale = p_scale
        self.p_rot = p_rot
        self.align_corners = align_corners
        
    def __call__(self, sample):
        x = sample['image']
        
        affines = []
        
        sz, sy, sx = 1,1,1
        if np.random.uniform() < self.p_scale:
            sz = np.random.uniform(*self.scale_z)
        if np.random.uniform() < self.p_scale:
            sy = np.random.uniform(*self.scale_y)
        if np.random.uniform() < self.p_scale:
            sx = np.random.uniform(*self.scale_x)
        affines.append(matrix_scale3d((sz,sy,sx)))
        
        if np.random.uniform() < self.p_rot:
            degz = np.random.uniform(*self.rot)
            affines.append(matrix_rotation3d_z(degz))
        if np.random.uniform() < self.p_rot:
            degy = np.random.uniform(*self.rot)
            affines.append(matrix_rotation3d_y(degy))
        if np.random.uniform() < self.p_rot:
            degx = np.random.uniform(*self.rot)
            affines.append(matrix_rotation3d_x(degx))

        affine_matrix = matrix_identity3d()
        for aff in affines:
            affine_matrix = np.dot(aff, affine_matrix)
    
        affgrid = F.affine_grid(torch.from_numpy(affine_matrix[:-1]).type(x.dtype).unsqueeze(0), x.unsqueeze(0).shape, align_corners=self.align_corners)
        affgrid = affgrid.to(x.device)
        x = F.grid_sample(x.unsqueeze(0), affgrid, align_corners=self.align_corners).squeeze(0)
        del affgrid
        
        return_items = {'image': x}

        if 'label' in sample.keys():
            y = sample['label']
            affgrid = F.affine_grid(torch.from_numpy(affine_matrix[:-1]).type(y.dtype).unsqueeze(0), y.unsqueeze(0).shape, align_corners=self.align_corners)
            affgrid = affgrid.to(x.device)
            y = F.grid_sample(y.unsqueeze(0), affgrid, align_corners=self.align_corners).squeeze(0)
            del affgrid
            return_items['label'] = y
        return return_items 
        
class RandomCrop3D(object):
    '''
    Random crop
    Args:
        crop_size: ZYX
    '''

    def __init__(self, crop_size):
        self.z_crop, self.y_crop, self.x_crop = crop_size
        
    def __call__(self, sample):
        x = sample['image']
        
        iz, iy, ix = x.shape[-3:]    
        zmin = random.randint(0, np.maximum(0, iz - self.z_crop))
        ymin = random.randint(0, np.maximum(0, iy - self.y_crop))
        xmin = random.randint(0, np.maximum(0, ix - self.x_crop))
        zmax = zmin + self.z_crop
        ymax = ymin + self.y_crop
        xmax = xmin + self.x_crop
            
        slicer = ((slice(zmin,zmax),slice(ymin,ymax),slice(xmin,xmax)))
        return_items = {'image': x[(slice(None),)*max(0,len(x.shape)-3)+slicer]}
                    
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y[(slice(None),)*max(0,len(y.shape)-3)+slicer]
        return return_items
    
class RandomMirror3D(object):
    '''
    Mirror x,y,z
    Args:
        axes: 0,1,2 = z,y,x
        p: apply if np.random.uniform() < p * 0.5
    '''
    def __init__(self, axes=(0,1,2), p=1):
        self.axes = axes
        self.p = p
        
    def __call__(self, sample):
        x = sample['image']
            
        slices = [0,0,0]
        for i in self.axes:
            if np.random.uniform() < self.p * 0.5:
                slices[i] = 1
        slices = [0] + slices
        flip_axes = tuple(np.argwhere(slices)[:,0])
        
        if len(flip_axes) > 0:
            if isinstance(x, Tensor):
                return_items = {'image': torch.flip(x, flip_axes)}   
            else:
                return_items = {'image': np.flip(x, flip_axes)}
        else:
            return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            if len(flip_axes) > 0:
                if isinstance(y, Tensor):
                    return_items = {'label': torch.flip(y, flip_axes)}   
                else:
                    return_items = {'label': np.flip(y, flip_axes)}
            else:
                return_items = {'label': y}            
        return return_items
    
class NoiseGaussian(object):
    '''
    Gaussian noise
    Args:
        noise_variance: range of noise variance in np.random.normal(0, variance)
        p: apply if np.random.uniform() < p
    '''    
    def __init__(self, noise_variance=(0,0.1), p=1):
        self.p = p
        self.noise_variance = noise_variance
        
    def __call__(self, sample):
        x = sample['image']
        
        for c in range(x.shape[0]):
            if np.random.uniform() < self.p:
                varmin, varmax = self.noise_variance
                if varmin == varmax:
                    variance = varmin
                else:
                    variance = random.uniform(varmin, varmax)
                if isinstance(x, Tensor):
                    x[c] += torch.normal(0.0, variance, size=x[c].shape, device=x.device)
                else:
                    x[c] += np.random.normal(0.0, variance, size=x[c].shape)
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items   
        
class AugmentBrightnessMultiplicative(object):
    '''
    Multiplicative augmentation on brightness
    Args:
        factor_range
        p: apply if np.random.uniform() < p
    '''    
    def __init__(self, factor_range=(0.75, 1.25), p=0.5):
        self.p = p
        self.factor_range = factor_range
        
    def __call__(self, sample):
        x = sample['image']
        
        for c in range(x.shape[0]):
            if np.random.uniform() < self.p:
                factor = np.random.uniform(*self.factor_range)
                x[c] *= factor
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items 
        
# almost same as multiplicative brightness for normalized input
class AugmentContrast(object):
    '''
    Augmentation on contrast
    Args:
        factor_range
        p: apply if np.random.uniform() < p
    '''    
    def __init__(self, factor_range=(0.75, 1.25), p=0.15, preserve_range=True):
        self.p = p
        self.factor_range = factor_range
        self.preserve_range = preserve_range
        
    def __call__(self, sample):
        x = sample['image']
        
        for c in range(x.shape[0]):
            if np.random.uniform() < self.p:                
                if np.random.random() < 0.5 and self.factor_range[0] < 1:
                    factor = np.random.uniform(self.factor_range[0], 1)
                else:
                    factor = np.random.uniform(max(self.factor_range[0], 1), self.factor_range[1])

                mn = x[c].mean()
                if self.preserve_range:
                    minm = x[c].min()
                    maxm = x[c].max()

                x[c] = (x[c] - mn) * factor + mn

                if self.preserve_range:
                    x[c][x[c] < minm] = minm
                    x[c][x[c] > maxm] = maxm
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items 


############################################################
## Basic Transforms
############################################################    

class ClipHU(object):
    def __init__(self, window_width, window_level):
        self.clip_min = window_level - 0.5*window_width
        self.clip_max = window_level + 0.5*window_width
        
    def __call__(self, sample):
        x = sample['image']
    
        x[x<self.clip_min] = self.clip_min
        x[x>self.clip_max] = self.clip_max
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items
    
class AddHU(object):
    '''torch grid_sample zero-padding should be -1024HU
    '''
    def __init__(self, value=1024):
        self.value = value
        
    def __call__(self, sample):
        x = sample['image']
        x = x + self.value
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items
    
class Normalize(object):
    def __init__(self, global_mean=None, global_std=None):
        self.global_mean = global_mean
        self.global_std = global_std
        
    def __call__(self, sample):
        x = sample['image']
        
        if not self.global_mean or not self.global_std:
            dims = tuple(np.arange(len(x.shape))[1:])
            if isinstance(x, Tensor):
                std, mean = torch.std_mean(x, dim=dims, keepdim=True)
            else:
                mean = np.mean(x, axis=dims, keepdims=True)
                std = np.std(x, axis=dims, keepdims=True)
        else:
            mean, std = self.global_mean, self.global_std
        
        x = (x-mean)/(std + 1e-5)
        
        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y
        return return_items

class CreateOnehotLabel(object):
    '''
    if onehot, convert gt from shape (b, z, y, x) to one hot encoding (b, z, y, x, c)
    if no onehot, convert gt from shape (b, z, y, x) to shape (b, z, y, x, 1)
    '''
    def __init__(self, num_classes, onehot=True):
        self.num_classes = num_classes
        self.onehot = onehot

    def __call__(self, sample):
        return_items = {'image': sample['image']}
        
        if 'label' in sample.keys():
            y = sample['label']
            if self.onehot:
                if isinstance(y, Tensor):
                    label_dtype = torch.float32
                    onehot_label = torch.zeros((self.num_classes,) + tuple(y.shape), dtype=label_dtype)
                    # dummy label
                    if torch.max(y) >= self.num_classes:
                        onehot_label = onehot_label - 1
                    for i in range(self.num_classes):
                        onehot_label[i] = (y == i).type(label_dtype)
                else:
                    label_dtype = np.float32
                    onehot_label = np.zeros((self.num_classes,) + tuple(y.shape), dtype=label_dtype)
                    # dummy label
                    if y.max() >= self.num_classes:
                        onehot_label = onehot_label - 1
                    for i in range(self.num_classes):
                        onehot_label[i] = (y == i).astype(label_dtype)
            else:
                if isinstance(y, Tensor):
                    onehot_label = torch.unsqueeze(y, 0)
                else:
                    onehot_label = np.expand_dims(y, axis=0)
            
            return_items['label'] = onehot_label

        return return_items
    
class BackgroundLabelProcess(object):
    '''
    For float-type onehot label, change all-zero points to background label and suppress background to not lose foreground labels
    '''
    def __init__(self, num_classes: int, weight=1, eps=1e-7):
        self.num_classes = num_classes
        self.weight = [weight,] + [1,]*(num_classes-1)
        self.eps = eps

    def __call__(self, sample):
        return_items = {'image': sample['image']}
        
        if 'label' in sample.keys():
            y = sample['label']
            y[0] += self.eps
            w = torch.tensor(self.weight, device=y.device)
            wy = y * w[(...,) + (None,)*(len(y.shape)-1)]
            y = wy / wy.sum(0).unsqueeze(0)
            return_items['label'] = y

        return return_items

class ToTensor3D(object):
    def __init__(self):
        pass
    
    def __call__(self, sample):
        x = sample['image']
        X = torch.from_numpy(x).float().contiguous()
        return_items = {'image': X}
        
        if 'label' in sample.keys():
            y = sample['label']
            Y = torch.from_numpy(y).float().contiguous()
            return_items['label'] = Y
        
        return return_items
    
class Padding3D(object):
    '''
    Add padding
    Args:
        size: target size (z,y,x)
        padding_value: constant padding. number or ['min','max','mean']
        center: if True, center padding
    '''

    def __init__(self, size, padding_value=0, center=True):
        self.size = np.array(size)
        assert any(isinstance(padding_value, x) for x in [str, numbers.Number])
        self.pv = padding_value
        self.center = center
        
    def __call__(self, sample):
        x = sample['image']

        input_size = x.shape[-3:]
        pads = self.size - input_size
        pads[pads<0] = 0
        # center padding
        if self.center:
            pads = tuple([(a//2, a-a//2) for a in pads.astype(int)])           
        else:
            pads = tuple([(0, a) for a in pads.astype(int)])   
            
        if isinstance(self.pv, str):
            if self.pv.lower() == 'min':
                pv = x.min()
            elif self.pv.lower() == 'max':
                pv = x.max()
            else:
                pv = x.mean()
        else:
            pv = self.pv
        if isinstance(x, Tensor):
            x = torch.nn.functional.pad(x, tuple(np.array(pads[::-1]).ravel()), mode='constant', value=pv)
        else:
            x = np.pad(x, ((0,0),)*max(0,len(x.shape)-3) + pads, 'constant', constant_values=pv)

        return_items = {'image': x}
        
        if 'label' in sample.keys():
            y = sample['label']
            # label padding must be background
            if isinstance(x, Tensor):
                y = torch.nn.functional.pad(y, tuple(np.array(pads[::-1]).ravel()), mode='constant', value=0)
            else:
                y = np.pad(y, ((0,0),)*max(0,len(y.shape)-3) + pads, 'constant', constant_values=0)
            return_items['label'] = y

        return return_items
    
class CenterCrop3D(object):
    '''
    Center crop
    Args:
        crop_size: ZYX
    '''

    def __init__(self, crop_size):
        self.z_crop, self.y_crop, self.x_crop = crop_size
        
    def __call__(self, sample):
        x = sample['image']
        
        iz, iy, ix = x.shape[-3:]    
        zmin = iz//2 - self.z_crop//2
        ymin = iy//2 - self.y_crop//2
        xmin = ix//2 - self.x_crop//2
        zmax = zmin + self.z_crop
        ymax = ymin + self.y_crop
        xmax = xmin + self.x_crop
            
        slicer = ((slice(zmin,zmax),slice(ymin,ymax),slice(xmin,xmax)))
        return_items = {'image': x[(slice(None),)*max(0,len(x.shape)-3)+slicer]}
                    
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y[(slice(None),)*max(0,len(y.shape)-3)+slicer]
            
        return return_items
    
class Crop3D(object):
    '''
    Crop
    Args:
        z_min: int, y_min: int, x_min: int, z_max: int, y_max: int, x_max: int
    '''

    def __init__(self, z_min: int, y_min: int, x_min: int, z_max: int, y_max: int, x_max: int):
        assert (z_min < z_max)&(y_min < y_max)&(x_min < x_max)
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        
    def __call__(self, sample):
        x = sample['image']
        slicer = ((slice(self.zmin,self.zmax),slice(self.ymin,self.ymax),slice(self.xmin,self.xmax)))
        return_items = {'image': x[(slice(None),)*max(0,len(x.shape)-3)+slicer]}
        
        if 'label' in sample.keys():
            y = sample['label']
            return_items['label'] = y[(slice(None),)*max(0,len(y.shape)-3)+slicer]
            
        return return_items
