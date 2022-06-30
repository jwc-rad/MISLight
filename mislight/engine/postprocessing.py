import glob
import json
import numpy as np
import os
import random
import SimpleITK as sitk
from tqdm.autonotebook import tqdm

from mislight.utils.image_resample import TorchResample, SkimageResample, ScipyResample
from mislight.utils.largest_cc import extract_topk_largest_candidates

class MyPostprocessing(object):
    def __init__(self, opt):
        self.opt = opt
        with open(opt.dataset_json, 'r') as f:
            self.ds_info = json.load(f)   
            
        self.datadir = opt.datadir
        self.result_dir = opt.result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.num_classes = opt.num_classes
        self.num_processes = opt.num_threads
        self.ipl_order_mask = opt.ipl_order_mask
        self.device = 'cpu'
        if opt.gpu_ids:
            try:
                self.device = 'cuda:'+str(opt.gpu_ids[0])
            except:
                pass
        self.coarse_export = opt.coarse_export
        if not opt.no_largest_cc:
            # keep only 1 largest CC per class
            self.cc_topk = [1,] * self.num_classes
        else:
            self.cc_topk = None
            
        self.keys = sorted([os.path.basename(x).split('.npy')[0] for x in glob.glob(os.path.join(self.datadir, '*.npy'))])
            
    def pp_single(self, key):
        resample_method = self.ds_info['resample_method']
        
        if resample_method.lower() == 'torch':
            resample_ = TorchResample        
        elif resample_method.lower() == 'skimage':
            resample_ = SkimageResample
        elif resample_method.lower() == 'scipy':
            resample_ = ScipyResample

        fpred = np.load(os.path.join(self.datadir, f'{key}.npy'))
        
        ## if coarse_export, postprocess larget_cc only and save npy
        if self.coarse_export:
            seg_pred = fpred.argmax(0).astype('uint8')
            if self.cc_topk:
                seg_pred = extract_topk_largest_candidates(seg_pred, self.cc_topk)
            tgt_file = os.path.join(self.result_dir, f"{key}.npy")
            np.save(tgt_file, seg_pred)       
            print(f'saved {tgt_file}')            
        else:        
            dir_source = self.ds_info['tgtdir']
            tsource = np.load(os.path.join(dir_source, f'{key}.npy'), allow_pickle=True)[()]
            source_size = tsource['source_size']
            padding = tsource['padding']            
            itk_spacing = tsource['source_itk_spacing'][0]
            itk_origin = tsource['source_itk_origin'][0]
            itk_direction = tsource['source_itk_direction'][0]
            del tsource
                
            # depad - resize
            padding = tuple(tuple(x) for x in np.array(padding).reshape(2, -1).T)
            resample_size = [s - sum(p) for s,p in zip(source_size, padding)]
            seg_pred, _ = resample_.resample_mask_to_size(fpred, resample_size, self.num_classes, self.ipl_order_mask, self.device, argmax=True)

            # largest CC
            seg_pred = seg_pred.astype('uint8')
            if self.cc_topk:
                seg_pred = extract_topk_largest_candidates(seg_pred, self.cc_topk)
                
            # repad
            seg_pred = np.pad(seg_pred, padding, 'constant', constant_values=0)
        
            # reorient RAI to original orientation. numpy array is (z,y,x)
            orient_x = itk_direction[0]
            orient_y = itk_direction[4]
            orient_z = itk_direction[8]
            if orient_z < 0:
                seg_pred = seg_pred[::-1]
            if orient_y < 0:
                seg_pred = seg_pred[:,::-1]
            if orient_x < 0:
                seg_pred = seg_pred[:,:,::-1]

            tgt_image = sitk.GetImageFromArray(seg_pred)
            tgt_image.SetSpacing(itk_spacing)
            tgt_image.SetOrigin(itk_origin)
            tgt_image.SetDirection(itk_direction)
            tgt_file = os.path.join(self.result_dir, f"{key}.{self.ds_info['file_extension']}")
            sitk.WriteImage(tgt_image, tgt_file)
            print(f'saved {tgt_file}')
        
    def run(self):
        for key in tqdm(self.keys):
            self.pp_single(key)  
