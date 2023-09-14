import copy
import os
import numpy as np
from PIL import Image
from typing import Optional, Sequence, List, Tuple, Union
import wandb

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

from mislight.utils.mask_process import postprocess_view

# save results callback
class ResultsCallback(pl.Callback):
    '''Save results to SAVE_DIR
    '''
    def __init__(
        self,
        result_dir: Union[List[str], str] = './results',
        save_npy = True,
        save_png = True,
        postprocess = False,
    ):
        super().__init__()
        if not isinstance(result_dir, list):
            result_dir = [result_dir]
        self.result_dir = result_dir
        self.save_npy = save_npy
        self.save_png = save_png
        self.postprocess = postprocess
    
    def _result_batch(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        keys = batch['metadata']['key']
        outs = pl_module.outputs.detach().cpu()
        
        original_shapes = batch['image_meta_dict']['spatial_shape']
        # resize_shape, original_shape
        if 'foreground_end_coord' in batch.keys() and 'foreground_start_coord' in batch.keys():
            resize_shapes = batch['foreground_end_coord'] - batch['foreground_start_coord']
            pads = torch.stack(
                [
                    torch.cat([torch.tensor([j1, j2]) for j1, j2 in zip(i1, i2)][::-1], dim=0) 
                    for i1, i2 in zip(batch['foreground_start_coord'], original_shapes - batch['foreground_end_coord'])
                ]
            )
        else:
            resize_shapes = original_shapes
            pads = torch.tensor([[0, 0, 0, 0]]*outs.shape[0])

        resize_shapes = resize_shapes.detach().cpu().numpy()
        original_shapes = original_shapes.detach().cpu().numpy()
        pads = pads.detach().cpu().numpy()
                
        for i in range(len(outs)):
            fn = keys[i]
            out = F.pad(
                F.interpolate(outs[i:i+1], size=tuple(resize_shapes[i]), mode='bilinear'),
                tuple(pads[i])
            )[0].numpy().transpose(0,2,1).astype(np.float16)
            
            if self.save_npy:
                npy_dir = os.path.join(self.result_dir[dataloader_idx], 'npy')
                os.makedirs(npy_dir, exist_ok=True)  
                npy_path = os.path.join(npy_dir, f'{fn}.npy')  
                np.save(npy_path, out)
                
            if self.save_png:
                png_dir = os.path.join(self.result_dir[dataloader_idx], 'png')
                os.makedirs(png_dir, exist_ok=True)
                png_path = os.path.join(png_dir, f'{fn}.png')
                
                n_png = out.argmax(0)
                if self.postprocess:
                    _, view = fn.split('_')
                    n_png = postprocess_view(n_png, view) 
                im = Image.fromarray(n_png.astype('uint8'))
                im.save(png_path)                           
    
    # Callback method
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._result_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._result_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
        
# save results callback
# npy save as non-resized, non-transposed npy with metadata
# png save as full
class SimpleResultsCallback(pl.Callback):
    '''Save results to SAVE_DIR
    '''
    def __init__(
        self,
        result_dir: Union[List[str], str] = './results',
        save_npy = True,
        save_png = True,
        postprocess = False,
    ):
        super().__init__()
        if not isinstance(result_dir, list):
            result_dir = [result_dir]
        self.result_dir = result_dir
        self.save_npy = save_npy
        self.save_png = save_png
        self.postprocess = postprocess
    
    def _result_batch(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        keys = batch['metadata']['key']
        outs = pl_module.outputs.detach().cpu()
        
        original_shapes = batch['image_meta_dict']['spatial_shape']
        # resize_shape, original_shape
        if 'foreground_end_coord' in batch.keys() and 'foreground_start_coord' in batch.keys():
            resize_shapes = batch['foreground_end_coord'] - batch['foreground_start_coord']
            pads = torch.stack(
                [
                    torch.cat([torch.tensor([j1, j2]) for j1, j2 in zip(i1, i2)][::-1], dim=0) 
                    for i1, i2 in zip(batch['foreground_start_coord'], original_shapes - batch['foreground_end_coord'])
                ]
            )
        else:
            resize_shapes = original_shapes
            pads = torch.tensor([[0, 0, 0, 0]]*outs.shape[0])

        resize_shapes = resize_shapes.detach().cpu().numpy()
        original_shapes = original_shapes.detach().cpu().numpy()
        pads = pads.detach().cpu().numpy()
                
        for i in range(len(outs)):
            fn = keys[i]
            if self.save_npy:
                npy_dir = os.path.join(self.result_dir[dataloader_idx], 'npysimple')
                os.makedirs(npy_dir, exist_ok=True)  
                npy_path = os.path.join(npy_dir, f'{fn}.npy')  
                
                n_npy = outs[i].numpy()
                savenpy = {}
                savenpy['npy'] = n_npy.astype(np.float16)
                savenpy['resize_shape'] = resize_shapes[i]
                savenpy['original_shape'] = original_shapes[i]
                savenpy['pad'] = pads[i]
                np.save(npy_path, savenpy)
                
            if self.save_png:
                png_dir = os.path.join(self.result_dir[dataloader_idx], 'png')
                os.makedirs(png_dir, exist_ok=True)
                png_path = os.path.join(png_dir, f'{fn}.png')
                
                out = F.pad(
                            F.interpolate(outs[i:i+1], size=tuple(resize_shapes[i]), mode='bilinear'),
                            tuple(pads[i])
                        )[0].numpy().transpose(0,2,1).astype(np.float16)
                
                n_png = out.argmax(0)
                if self.postprocess:
                    _, view = fn.split('_')
                    n_png = postprocess_view(n_png, view) 
                im = Image.fromarray(n_png.astype('uint8'))
                im.save(png_path)                            
    
    # Callback method
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._result_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._result_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
        
# save results as bounding box binary callback
class BboxResultsCallback(pl.Callback):
    '''Save results to SAVE_DIR
    '''
    def __init__(
        self,
        result_dir: Union[List[str], str] = './results',
    ):
        super().__init__()
        if not isinstance(result_dir, list):
            result_dir = [result_dir]
        self.result_dir = result_dir
    
    def _result_batch(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        keys = batch['metadata']['key']
        outs = pl_module.outputs.detach().cpu()

        original_shapes = batch['image_meta_dict']['spatial_shape']
        # resize_shape, original_shape
        if 'foreground_end_coord' in batch.keys() and 'foreground_start_coord' in batch.keys():
            resize_shapes = batch['foreground_end_coord'] - batch['foreground_start_coord']
            pads = torch.stack(
                [
                    torch.cat([torch.tensor([j1, j2]) for j1, j2 in zip(i1, i2)][::-1], dim=0) 
                    for i1, i2 in zip(batch['foreground_start_coord'], batch['foreground_start_coord'])
                ]
            )
        else:
            resize_shapes = original_shapes
            pads = torch.tensor([[0, 0, 0, 0]]*outs.shape[0])
        resize_shapes = resize_shapes.detach().cpu().numpy()
        original_shapes = original_shapes.detach().cpu().numpy()
        pads = pads.detach().cpu().numpy()
        
        outs = [outs[i].numpy().argmax(0) for i in range(outs.shape[0])]
        bbox = []
        for out in outs:
            get_pos = np.argwhere(out > 0)
            xmin, ymin = np.min(get_pos, axis=0)
            xmax, ymax = np.max(get_pos, axis=0)
            bbox.append([xmin/out.shape[0], ymin/out.shape[1], xmax/out.shape[0], ymax/out.shape[1]])
        
        bbox_resize = []
        for i, b in enumerate(bbox):
            xsh, ysh = resize_shapes[i]
            bbox_resize.append([int(b[0]*xsh), int(b[1]*ysh), int(b[2]*xsh), int(b[3]*ysh)])
        bbox_resize = np.array(bbox_resize)        
        bbox_resize_pad = (bbox_resize + pads).astype(int)

        for i in range(bbox_resize_pad.shape[0]):
            fn = keys[i]
            png_dir = os.path.join(self.result_dir[dataloader_idx], 'bbox')
            os.makedirs(png_dir, exist_ok=True)
            png_path = os.path.join(png_dir, f'{fn}.png')
            
            n_png = np.zeros(tuple(original_shapes[i]))
            xmin, ymin, xmax, ymax = bbox_resize_pad[i]
            n_png[xmin:xmax, ymin:ymax] = 1
            
            n_png = n_png.T
            im = Image.fromarray(n_png.astype('uint8'))
            im.save(png_path)        
                
    # Callback method
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._result_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0)
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._result_batch(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) 
        
            
            
# metrics callback
# for logging best validation metric (best means when monitor is best. e.g. val_loss)
class MetricsBestValidCallback(pl.Callback):
    def __init__(
        self,
        metric: Union[List[str], str] = 'metric/val_mIOU',
        monitor: str = 'loss/val_loss',
        monitor_mode: str = 'min',
    ):
        super().__init__()
        if not isinstance(metric, list):
            metric = [metric]
        self.metric = metric
        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.all_metrics = {}
    
    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        for k,v in each_me.items():
            if k in self.all_metrics.keys():
                self.all_metrics[k].append(v.item())
            else:
                self.all_metrics[k] = [v.item()]                
                
        # last validation epoch
        every_n_epoch = pl_module.hparams['opt'].check_val_every_n_epoch
        if pl_module.current_epoch == every_n_epoch*(trainer.max_epochs//every_n_epoch) - 1:
            self.log_best()
                
    def log_best(self):
        monitored = self.all_metrics[self.monitor]
        if self.monitor_mode == 'min':
            idx = np.argmin(np.array(monitored))
        else:
            idx = np.argmax(np.array(monitored))
        
        for k in self.metric:
            self.log(k+'_best', self.all_metrics[k][idx])