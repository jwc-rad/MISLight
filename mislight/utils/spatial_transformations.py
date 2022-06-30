import math
import numbers
import numpy as np
from typing import Iterable

import torch
import torch.nn.functional as F

from .image_resample import order2mode

def generate_deformation_grid(shape, num_points=16, sigma_factor=(0.75, 1), truncate=4, order=1, align_corners=True, device=None):
    '''shape: B,C,H,W(,D)
    '''
    batch = shape[0]
    img_shape = shape[2:]
    dim = len(img_shape)
    grid_shape = (num_points,) * dim

    sigma_unit = num_points / truncate / 2
    sigma_range = (sigma_unit * sigma_factor[0], sigma_unit * sigma_factor[1])
    sigmas = (sigma_range[1] - sigma_range[0])*torch.rand(batch, device=device) + sigma_range[0]

    kernel_sizes = 2*(4*sigmas + 0.5).long() + 1
    kernel_sizes = torch.stack([kernel_sizes]*dim, axis=1)
    sigmas = torch.stack([sigmas]*dim, axis=1)

    grids = []
    for kernel_size, sigma in zip(kernel_sizes, sigmas):
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, device=device)
                for size in kernel_size
            ],
            indexing='ij',
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())

        grid = []
        for _ in range(dim):
            tx = torch.rand((1,1) + grid_shape, device=device)*2 - 1
            tx = F.pad(tx, (kernel_size[0].item()//2,)*2*dim, mode='constant', value=0)
            tx = F.conv3d(tx, weight=kernel)
            grid.append(tx)
        grid = torch.cat(grid, 1)
        grid = F.interpolate(grid, size=img_shape, mode='trilinear', align_corners=align_corners)
        grid = grid.permute(0,2,3,4,1)
        grids.append(grid)
    grids = torch.cat(grids, 0)

    # now add to identity grid
    agrid = F.affine_grid(torch.stack([tensormat_identity3d(device)[:-1]]*batch), shape, align_corners=align_corners)

    deform = agrid + (grids * sigmas[(slice(None),) + (None,)*dim])

    # rescale to [-1, 1]
    deform -= deform.view(batch,-1,dim).min(1, True)[0].view(batch, *(1,)*dim, dim)
    deform /= deform.view(batch,-1,dim).max(1, True)[0].view(batch, *(1,)*dim, dim)
    deform = deform * 2 -1
    
    return deform

# affine - z, y, x order Tensor
def tensormat_identity3d(device=None):
    M = torch.tensor([
        [         1,         0,         0,         0],
        [         0,         1,         0,         0],
        [         0,         0,         1,         0],
        [         0,         0,         0,         1],
    ], dtype=torch.float32, device=device)
    return M    

def tensormat_rotation3d_z(deg, device=None):
    t = deg/180*np.pi
    M = torch.tensor([
        [         1,         0,         0,         0],
        [         0, np.cos(t), np.sin(t),         0],
        [         0,-np.sin(t), np.cos(t),         0],
        [         0,         0,         0,         1],
    ], dtype=torch.float32, device=device)
    return M

def tensormat_rotation3d_y(deg, device=None):
    t = deg/180*np.pi
    M = torch.tensor([
        [ np.cos(t),         0,-np.sin(t),         0],
        [         0,         1,         0,         0],
        [ np.sin(t),         0, np.cos(t),         0],
        [         0,         0,         0,         1],
    ], dtype=torch.float32, device=device)
    return M

def tensormat_rotation3d_x(deg, device=None):
    t = deg/180*np.pi
    M = torch.tensor([
        [ np.cos(t),-np.sin(t),         0,         0],
        [ np.sin(t), np.cos(t),         0,         0],
        [         0,         0,         1,         0],
        [         0,         0,         0,         1],
    ], dtype=torch.float32, device=device)
    return M

def tensormat_scale3d(s, device=None):
    if isinstance(s, Iterable):
        assert len(s)==3
        sz, sy, sx = s
    else:
        sz = sy = sx = s
    M = torch.tensor([
        [        sz,         0,         0,         0],
        [         0,        sy,         0,         0],
        [         0,         0,        sx,         0],
        [         0,         0,         0,         1],
    ], dtype=torch.float32, device=device)
    return M    

# affine - z, y, x order
def matrix_identity3d():
    M = np.array([
        [         1,         0,         0,         0],
        [         0,         1,         0,         0],
        [         0,         0,         1,         0],
        [         0,         0,         0,         1],
    ], dtype=np.float32)
    return M    

def matrix_rotation3d_z(deg):
    t = deg/180*np.pi
    M = np.array([
        [         1,         0,         0,         0],
        [         0, np.cos(t), np.sin(t),         0],
        [         0,-np.sin(t), np.cos(t),         0],
        [         0,         0,         0,         1],
    ], dtype=np.float32)
    return M

def matrix_rotation3d_y(deg):
    t = deg/180*np.pi
    M = np.array([
        [ np.cos(t),         0,-np.sin(t),         0],
        [         0,         1,         0,         0],
        [ np.sin(t),         0, np.cos(t),         0],
        [         0,         0,         0,         1],
    ], dtype=np.float32)
    return M

def matrix_rotation3d_x(deg):
    t = deg/180*np.pi
    M = np.array([
        [ np.cos(t),-np.sin(t),         0,         0],
        [ np.sin(t), np.cos(t),         0,         0],
        [         0,         0,         1,         0],
        [         0,         0,         0,         1],
    ], dtype=np.float32)
    return M

def matrix_scale3d(s):
    if isinstance(s, Iterable):
        assert len(s)==3
        sz, sy, sx = s
    else:
        sz = sy = sx = s
    M = np.array([
        [        sz,         0,         0,         0],
        [         0,        sy,         0,         0],
        [         0,         0,        sx,         0],
        [         0,         0,         0,         1],
    ], dtype=np.float32)
    return M    
