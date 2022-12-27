import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import re

###############################################################################
# File and Folder Managers
###############################################################################

def make_dir_with_number(savedir, fname=None, num=None, digits=5):
    os.makedirs(savedir, exist_ok=True)
    currentlist = [x for x in os.listdir(savedir) if os.path.isdir(os.path.join(savedir,x))]
    if num is None:
        numbers = []
        for x in currentlist:
            x = x.split('_')[0]
            try:
                numbers.append(int(x))
            except:
                continue
        if len(numbers)==0:
            N = 0
        else:
            N = max(numbers)
    else:
        N = num-1
        
    newname_list = [f'{N+1:0{digits}d}']
    if fname:
        newname_list.append(fname)
    newname = os.path.join(savedir, '_'.join(newname_list))
    os.makedirs(newname, exist_ok=True)
    return newname

def find_dir_with_number(foldernum, basedir):
    list_folders = [x for x in os.listdir(basedir) if (~x.startswith('.'))&(os.path.isdir(os.path.join(basedir,x)))]
    list_foldernum = [int(x.split('_')[0]) for x in list_folders]
    if foldernum in list_foldernum:
        return os.path.join(basedir, list_folders[list_foldernum.index(foldernum)])
    else:
        print(f'no {foldernum} in {basedir}')
        return None

def find_files(case, srcdir, prefix='', suffix=''):
    find = []
    for x in glob.glob(os.path.join(srcdir,'*.*')):
        if prefix+case+suffix in os.path.basename(x):
            find.append(x)
    return sorted(find)

def find_file(case, srcdir, prefix='', suffix='', fileonly=True):
    find = None
    if fileonly:
        querypath = os.path.join(srcdir,'*.*')
    else:
        querypath = os.path.join(srcdir,'*')
    for x in glob.glob(querypath):
        if prefix+case+suffix in os.path.basename(x):
            find = x
            return find
    return find
    
###############################################################################
# Image Processing
###############################################################################

def rescale_percent(x, percent=[10,99.9]):
    m, M = np.percentile(x, percent)
    if np.abs(M-m)<1e-5:
        xx = np.zeros(x.shape, dtype=np.float32)
    else:
        xx = (x.astype(np.float32)-m)/(M-m)
        xx[xx>1] = 1
        xx[xx<0] = 0
    return xx

def rescale_minmax(X):
    X = X.astype(np.float32)
    m = np.min(X)
    M = np.max(X)
    if np.abs(M-m)<1e-5:
        R = np.zeros(X.shape, dtype=X.dtype)
    else:
        R = (X-m)/(M-m)
    R[R<0] = 0
    R[R>1] = 1
    return R
    
###############################################################################
# Misc.
###############################################################################

def safe_repeat(x, n):
    if not isinstance(x, (list, tuple)):
        x = [x] * n
    else:
        assert len(x) == n
    return x

def label2colormap(x):
    palette = [
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [255,255,0],
        [0,255,255],
        [255,0,255],
        [255,239,213],
        [0,0,205],
        [205,133,63],
        [210,180,140],
        [102,205,170],
        [0,0,128],
        [0,139,139],
    ]
    
    if x.shape[-1]==-1:
        newsh = x.shape[:-1] + (3,)
    else:
        newsh = x.shape + (3,)
    
    cmap = np.zeros(newsh, dtype='uint8')
    N = np.max(x)
    for i in range(1,1+int(N)):
        cmap[x==i] = palette[i-1]
    return cmap

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def mydisplay(images, cols=2, figsize=7, figratio=1, showz=0.5, zmaxis=None, window=None, colorbar=False, gray=True):
    r = int(np.ceil(len(images)/cols))
    c = cols
    plt.style.use('default')
    if c>1:
        fig, axs = plt.subplots(r,c, figsize=(figsize*c,figsize*r*figratio))
    else:
        fig, axs = plt.subplots(r, figsize=(figsize*c,figsize*r*figratio))
    
    for i,img in enumerate(images):
        if r==1:
            idx = i%c
        elif c==1:
            idx = i%r
        else:
            idx = (i//c,i%c)
        if window:
            if gray:
                fig0 = axs[idx].imshow(img, cmap='gray', vmin=window[0], vmax=window[1])
            else:
                fig0 = axs[idx].imshow(img, vmin=window[0], vmax=window[1])
        else:
            if gray:
                fig0 = axs[idx].imshow(img, cmap='gray')
            else:
                fig0 = axs[idx].imshow(img)
        if colorbar:
            fig.colorbar(fig0, ax=axs[idx])
        if zmaxis:
            axs[idx].axis(zmaxis)

    plt.show()
    plt.close(fig)