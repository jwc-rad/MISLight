import glob
import json
import numpy as np
import os
import random
import SimpleITK as sitk
from tqdm.autonotebook import tqdm

from mislight.utils.image_resample import TorchResample

class MyPreprocessing(object):    
    def __init__(self, opt):
        self.opt = opt
        self.inference = opt.inference
        self.file_extension = opt.file_extension
        self.num_processes = opt.num_threads
        self.device = 'cpu'
        if opt.gpu_ids:
            try:
                self.device = 'cuda:'+str(opt.gpu_ids[0])
            except:
                pass
        self.ds = {}
        self.ds['file_extension'] = opt.file_extension
        self.ds['resample_method'] = opt.resample_method  
        self.ds['resample_target'] = opt.resample_target  
        self.ds['max_dataset_size'] = opt.max_dataset_size
        
        self.ds['srcdirimage'] = opt.datadir
        if opt.dir_image:
            self.ds['tgtdir'] = os.path.join(opt.save_dir, opt.dir_image)
        else:
            self.ds['tgtdir'] = opt.save_dir          
        if not self.inference:
            self.ds['srcdirlabel'] = os.path.join(opt.dataroot, opt.dir_label)
        
        # for pre-cropping
        self.ds['previousdir'] = opt.dir_previous
        self.padding_buffer = opt.padding_buffer
        self.input_min_size = np.array(opt.input_min_size)
        
        if opt.resample_method.lower() == 'torch':
            self.resample = TorchResample
            assert opt.ipl_order_image in [0,1]
            assert opt.ipl_order_mask in [0,1]
        else:
            raise NotImplementedError(f'Resample method [{opt.resample_method}] is not recognized')
        self.ds['ipl_order_image'] = self.ipl_order_image = opt.ipl_order_image
        self.ds['ipl_order_mask'] = self.ipl_order_mask = opt.ipl_order_mask
            
        self.resample_target = opt.resample_target.lower()
        assert self.resample_target in ['spacing', 'size']
        
        if (opt.resample_fixed) and (self.resample_target == 'size'):
            self.resample_fixed = np.array(opt.resample_fixed).astype(int).tolist()
        else:
            self.resample_fixed = opt.resample_fixed
            
    def save_dataset(self):
        savepath = os.path.join(self.opt.save_dir, 'dataset.json')
        with open(savepath, 'w') as f:
            json.dump(self.ds, f)
            
    def check_data(self, check_resample_target=True):
        ## basic check        
        images = sorted(glob.glob(os.path.join(self.ds['srcdirimage'], f'*.{self.file_extension}')))
        image_names = [os.path.basename(x) for x in images]
        image_channels = sorted(set(x.split(f'.{self.file_extension}')[0].split('_')[-1] for x in image_names))
        image_keys = sorted(set('_'.join(x.split('_')[:-1]) for x in image_names))
        nc_input = len(image_channels)
        
        assert len(images)==nc_input*len(image_keys)
        self.ds['nc_input'] = nc_input
        self.ds['nc_names'] = image_channels
        num_cases = min(self.ds['max_dataset_size'], len(image_keys)) 
        
        if not self.inference:
            labels = sorted(glob.glob(os.path.join(self.ds['srcdirlabel'], f'*.{self.file_extension}')))
            label_names = [os.path.basename(x) for x in labels] 
            label_keys = sorted(set(x.split('.'+self.file_extension)[0] for x in label_names))
            
            assert not bool([x for x in label_keys if not (x in image_keys)])
            image_keys = label_keys + [x for x in image_keys if not (x in label_keys)]
                       
        ## partial loading
        image_keys = image_keys[:num_cases]
        images = []
        for k in image_keys:
            for c in image_channels:
                images.append(os.path.join(self.ds['srcdirimage'], f'{k}_{c}.{self.file_extension}'))
        self.ds['image_keys'] = image_keys
        
        if not self.inference:          
            label_keys = label_keys[:num_cases]
            labels = [os.path.join(self.ds['srcdirlabel'], f'{x}.{self.file_extension}') for x in label_keys]
            self.ds['label_keys'] = label_keys
              
            label_classes = []
            for x in tqdm(labels):
                #timage = nib.load(x)
                #timg = timage.get_fdata().astype(timage.header.get_data_dtype())
                timg = sitk.GetArrayFromImage(sitk.ReadImage(x))
                for i in range(1, int(1+np.max(timg))):
                    if (i in timg) and (not i in label_classes):
                        label_classes.append(i)
            assert label_classes == list(range(1,1+max(label_classes))) # labels should be consecutive integers
            self.ds['num_classes'] = max(label_classes)
            print(f'number of label classes = {max(label_classes)}')   
            
        ## check median resample target
        if check_resample_target and self.resample_fixed:
            images0 = [x for x in images if os.path.basename(x).split('.'+self.file_extension)[0].split('_')[-1]==image_channels[0]]
            spacings = []
            sizes = []
            for x in tqdm(images0):
                timg = sitk.ReadImage(x)
                spacings.append(timg.GetSpacing()[::-1])
                sizes.append(timg.GetSize()[::-1])
                #timg = nib.load(x)
                #spacings.append(timg.header.get_zooms())
                #sizes.append(timg.header.get_data_shape())

            spacing_median = np.median(np.array(spacings), axis=0).tolist()
            size_median = np.median(np.array(sizes), axis=0).tolist()
            self.ds['median_source_spacing'] = spacing_median
            self.ds['median_source_size'] = size_median
            print(f'median source spacing = {spacing_median}')
            print(f'median source size = {size_median}')
            
            spacing_min = np.min(np.array(spacings), axis=0).tolist()
            size_min = np.min(np.array(sizes), axis=0).tolist()
            self.ds['min_source_spacing'] = spacing_min
            self.ds['min_source_size'] = size_min
            print(f'min source spacing = {spacing_min}')
            print(f'min source size = {size_min}')
            
            spacing_max = np.max(np.array(spacings), axis=0).tolist()
            size_max = np.max(np.array(sizes), axis=0).tolist()
            self.ds['max_source_spacing'] = spacing_max
            self.ds['max_source_size'] = size_max
            print(f'max source spacing = {spacing_max}')
            print(f'max source size = {size_max}')

        ## set resample target
        if self.resample_fixed:
            if self.resample_target == 'spacing':
                self.ds['target_spacing'] = self.resample_fixed
            elif self.resample_target == 'size':
                self.ds['target_size'] = self.resample_fixed         
        else:        
            if self.resample_target == 'spacing':
                self.ds['target_spacing'] = spacing_median
            elif self.resample_target == 'size':
                self.ds['target_size'] = size_median
        tgt_value = self.ds[f'target_{self.resample_target}']
        print(f'resample targeting {self.resample_target} to {tgt_value}')
            
        return True
    
    def resample_single_data(self, key, spacings, sizes):
        target_value = self.ds[f'target_{self.resample_target}']
        srcimg = self.ds['srcdirimage']
        previous = self.ds['previousdir']
        if not self.inference:
            srclbl = self.ds['srcdirlabel']
            
        timages = []
        itk_meta = {'spacing':[], 'origin': [], 'direction': []}
        for i in range(self.ds['nc_input']):
            nc_name = self.ds['nc_names'][i]
            tfile = os.path.join(srcimg, f'{key}_{nc_name}.{self.file_extension}')
            timage = sitk.ReadImage(tfile)
            
            # save metadata
            itk_meta['spacing'].append(timage.GetSpacing())
            itk_meta['origin'].append(timage.GetOrigin())
            itk_meta['direction'].append(timage.GetDirection())
            
            # set to RAI
            orient_x = timage.GetDirection()[0]
            orient_y = timage.GetDirection()[4]
            orient_z = timage.GetDirection()[8]
            if orient_x < 0:
                timage = timage[::-1]
            if orient_y < 0:
                timage = timage[:,::-1]
            if orient_z < 0:
                timage = timage[:,:,::-1]
            source_size = np.array(timage.GetSize()[::-1])  
            source_spacing = np.array(timage.GetSpacing()[::-1])  
            
            # get bounding box if previous mask is present
            timg = sitk.GetArrayFromImage(timage)
            bbox = np.array([[0,]*len(timg.shape), timg.shape])
            if previous:
                tprevfile = os.path.join(previous, f'{key}.npy')
                if os.path.exists(tprevfile):
                    tprevious = np.load(tprevfile)
                    if (len(tprevious.shape) > 3) and (tprevious.shape[0] > 1):
                        binary_mask = np.argmax(tprevious, axis=0) > 0
                    else:
                        binary_mask = np.squeeze(tprevious > 0)
                    bbox = get_new_bbox(binary_mask, source_size, self.input_min_size/source_spacing, self.padding_buffer)
            padding = np.abs(np.array([[0,]*len(timg.shape), timg.shape]) - bbox)
                
            slicer = ()
            for j in range(bbox.shape[-1]):
                slicer += (slice(bbox[0,j], bbox[1,j]),)
            timg = timg[slicer]
            
            # resampling
            if self.resample_target == 'spacing':
                if all(np.array(source_spacing) == np.array(target_value)):
                    timgresample = timg
                    zoom_factor = np.array([1,1,1])
                else:
                    timgresample, zoom_factor = self.resample.resample_to_spacing(timg, source_spacing, target_value, self.ipl_order_image, self.device)
            elif self.resample_target == 'size':
                if all(np.array(timg.shape) == np.array(target_value)):
                    timgresample = timg
                    zoom_factor = np.array([1,1,1])
                else:
                    timgresample, zoom_factor = self.resample.resample_to_size(timg, target_value, self.ipl_order_image, self.device)                
            timages.append(timgresample)
        timages = np.stack(timages, axis=0)

        if self.resample_target == 'spacing':
            tsave = {
                'image': timages,
                'target_spacing': target_value,
                'target_size': list(timages.shape[1:]),
                'source_spacing': source_spacing.tolist(),
                'source_size': source_size.tolist(),
                'padding': padding.ravel().tolist(),
                'source_orientation': [orient_z, orient_y, orient_x],
                'target_orientation': [1, 1, 1],
                'source_itk_spacing': itk_meta['spacing'],
                'source_itk_origin': itk_meta['origin'],
                'source_itk_direction': itk_meta['direction'],
            }
        elif self.resample_target == 'size':
            tsave = {
                'image': timages,
                'target_spacing': source_spacing * zoom_factor,
                'target_size': target_value,
                'source_size': source_size.tolist(),    
                'source_spacing': source_spacing.tolist(),
                'padding': padding.ravel().tolist(),
                'source_orientation': [orient_z, orient_y, orient_x],
                'target_orientation': [1, 1, 1],
                'source_itk_spacing': itk_meta['spacing'],
                'source_itk_origin': itk_meta['origin'],
                'source_itk_direction': itk_meta['direction'],
            }

        # process label for labeled cases in training
        if (not self.inference) and (key in self.ds['label_keys']):
            tfile = os.path.join(srclbl, f'{key}.{self.file_extension}')
            timage = sitk.ReadImage(tfile)
            #timage = nib.load(tfile)
            # set to RAI
            orient_x = timage.GetDirection()[0]
            orient_y = timage.GetDirection()[4]
            orient_z = timage.GetDirection()[8]
            if orient_x < 0:
                timage = timage[::-1]
            if orient_y < 0:
                timage = timage[:,::-1]
            if orient_z < 0:
                timage = timage[:,:,::-1]
            source_size = timage.GetSize()[::-1]  
            source_spacing = timage.GetSpacing()[::-1]  
            
            timg = sitk.GetArrayFromImage(timage)    
            bbox = np.array([[0,]*len(timg.shape), timg.shape])
            if previous:
                tprevious = np.load(os.path.join(previous, f'{key}.npy'))
                if (len(tprevious.shape) > 3) and (tprevious.shape[0] > 1):
                    binary_mask = np.argmax(tprevious, axis=0) > 0
                else:
                    binary_mask = np.squeeze(tprevious > 0)
                bbox = get_new_bbox(binary_mask, source_size, self.input_min_size/source_spacing, self.padding_buffer)
            padding = np.abs(np.array([[0,]*len(timg.shape), timg.shape]) - bbox)
                
            slicer = ()
            for j in range(bbox.shape[-1]):
                slicer += (slice(bbox[0,j], bbox[1,j]),)
            timg = timg[slicer]
            
            if self.resample_target == 'spacing':
                if all(np.array(timage.GetSpacing()[::-1]) == np.array(target_value)):
                    timgresample = timg
                else:
                    timgresample, _ = self.resample.resample_mask_to_spacing(timg, source_spacing, target_value, self.ds['num_classes'], self.ipl_order_mask, self.device)
            elif self.resample_target == 'size':
                if all(np.array(timg.shape) == np.array(target_value)):
                    timgresample = timg
                else:
                    timgresample, _ = self.resample.resample_mask_to_size(timg, target_value, self.ds['num_classes'], self.ipl_order_mask, self.device) 

            tsave['label'] = timgresample    
            
        spacings.append(tsave['target_spacing'])
        sizes.append(tsave['target_size'])
        
        # save
        newpath = os.path.join(self.ds['tgtdir'], f'{key}.npy')
        np.save(newpath, tsave)    
        return spacings, sizes
            
    def resample_data(self):
        print('resampling...')
        target_value = self.ds[f'target_{self.resample_target}']
        
        tgtdir = self.ds['tgtdir']
        os.makedirs(tgtdir, exist_ok=True)
            
        spacings = []
        sizes = []            
        for key in tqdm(self.ds['image_keys']):
            spacings, sizes = self.resample_single_data(key, spacings, sizes)
                                    
        spacing_median = np.median(np.array(spacings), axis=0).tolist()
        size_median = np.median(np.array(sizes), axis=0).astype(int).tolist()
        self.ds['median_target_spacing'] = spacing_median
        self.ds['median_target_size'] = size_median
        print(f'median target spacing = {spacing_median}')
        print(f'median target size = {size_median}')
        
        
def get_new_bbox(mask, source_size, input_min_size, padding = 1):
    binary_idx = np.argwhere(mask>0)
    if len(binary_idx) > 0:
        binary_mins = np.min(binary_idx, axis=0)/np.array(mask.shape)
        binary_maxs = (1+np.max(binary_idx, axis=0))/np.array(mask.shape)
        s = (binary_maxs - binary_mins)*padding
        binary_center = 0.5*(binary_mins + binary_maxs)
    else:
        s = input_min_size / source_size
        binary_center = np.array([0.5,]*len(mask.shape))
    newbbox = np.stack([binary_center - 0.5*s, binary_center + 0.5*s])
    newbbox = np.clip(newbbox, 0, 1)
    return np.rint(newbbox * source_size).astype(int)
