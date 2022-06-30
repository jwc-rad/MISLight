import json
import os
import numpy as np
from PIL import Image
import SimpleITK as sitk

from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

class MyTrainer(pl.Trainer):
    '''Pytorch Lightning Trainer with custom callbacks, logger, args, etc.
    '''
    
    def __init__(self, pl_module: pl.LightningDataModule, **kwargs):
        opt = pl_module.opt
        self.opt = opt
        self.keys_test = pl_module.keys_test
        nkwargs = {}
        nkwargs['default_root_dir'] = opt.run_base_dir
        nkwargs['gpus'] = opt.gpu_ids
        nkwargs['callbacks'] = self.define_callbacks(opt)
        nkwargs['logger'] = self.define_loggers(opt)
        # mixed precision is default
        nkwargs['precision'] = 32 if opt.single_precision else 16
        
        if not opt.inference:
            nkwargs['max_epochs'] = opt.n_epochs   
            nkwargs['log_every_n_steps'] = opt.log_every_n_steps
            nkwargs['detect_anomaly'] = opt.detect_anomaly
        
        # if any, update additional kwargs, but pl_module's opt has higher priority than additional kwargs
        for k,v in kwargs.items():
            if not k in nkwargs.keys():
                nkwargs[k] = v
        
        super().__init__(**nkwargs)
                
    def define_callbacks(self, opt):
        L = []
        cb_results = SegmentationResultsCallback(opt.result_dir)
        L.append(cb_results)
        if not opt.inference:
            # ModelCheckpoint
            if opt.checkpoint_nooverwrite:
                save_top_k = -1
            else:
                save_top_k = 1
            cb_checkpoint = ModelCheckpoint(
                dirpath=os.path.join(opt.save_dir, 'checkpoint'),
                every_n_epochs=opt.checkpoint_every_n_epochs,
                every_n_train_steps=opt.checkpoint_every_n_train_steps,
                filename=opt.checkpoint_filename,
                auto_insert_metric_name=True,
                save_weights_only=not opt.save_fullmodel,
                save_top_k=(-1 if opt.checkpoint_nooverwrite else 1),
            )
            L.append(cb_checkpoint)    
            cb_lrmonitor = LearningRateMonitor(logging_interval='epoch')
            L.append(cb_lrmonitor)
        return L
    
    def define_loggers(self, opt):
        if not opt.inference:
            save_dir_split = opt.save_dir.split(os.sep)
        else:
            save_dir_split = opt.result_dir.split(os.sep)
            
        save_dir_split = ['.','.'] + save_dir_split
        log_version = save_dir_split.pop()
        log_name = save_dir_split.pop()
        log_dir = os.path.join(*save_dir_split)

        L = []
        if opt.log_type:
            if 'csv' in opt.log_type.lower():        # CSV Logger       
                L.append(CSVLogger(log_dir, name=log_name, version=log_version))
            if 'tb' in opt.log_type.lower():
                L.append(TensorBoardLogger(log_dir, name=log_name, version=log_version, sub_dir='tensorboard'))
        
        return L  

class SegmentationResultsCallback(pl.Callback):
    '''
    Args:
    
    '''
    def __init__(
        self,
        result_dir: str = './results',
    ):
        super().__init__()
        self.result_dir = result_dir
        
        self.current_test_idx = 0
    
    # save results on change of current_test_idx
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.current_test_idx != dataloader_idx:
            self.save_results(trainer, pl_module)
            self.current_test_idx = dataloader_idx
            
    # save results for the last case
    def on_test_end(self, trainer, pl_module):
        self.save_results(trainer, pl_module)

    def save_results(self, trainer, pl_module):
        fpred = list(pl_module.get_current_results().values())[0]
        current_key = trainer.keys_test[self.current_test_idx]

        newpath = os.path.join(self.result_dir, f'{current_key}.npy')
        np.save(newpath, fpred)
        print(f'saved {newpath}')
