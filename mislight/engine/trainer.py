from imageio import imwrite
import json
import os
import numpy as np
from PIL import Image
from typing import Optional, Sequence, List, Tuple, Union
import wandb

import torch
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from .callbacks import ResultsCallback, BboxResultsCallback, SimpleResultsCallback, MetricsBestValidCallback

class MyTrainer(pl.Trainer):
    '''Pytorch Lightning Trainer with custom callbacks, logger, args, etc.
    '''
    
    def __init__(self, pl_module: pl.LightningDataModule, **kwargs):
        opt = pl_module.opt
        self.opt = opt
        self.inference = opt.inference

        nkwargs = {}
        nkwargs['default_root_dir'] = opt.run_base_dir
        if -1 in opt.gpu_ids:
            nkwargs['accelerator'] = 'cpu'
        else:
            nkwargs['accelerator'] = 'gpu'
            nkwargs['devices'] = opt.gpu_ids
            
        
        nkwargs['logger'] = self.define_loggers(opt)
        nkwargs['log_every_n_steps'] = opt.log_every_n_steps
        nkwargs['callbacks'] = self.define_callbacks(opt)
        # mixed precision is default
        nkwargs['precision'] = 16 if opt.mixed_precision else 32
        
        if not self.inference:
            nkwargs['max_epochs'] = opt.max_epochs   
            nkwargs['check_val_every_n_epoch'] = opt.check_val_every_n_epoch
            nkwargs['detect_anomaly'] = opt.detect_anomaly
        
        # if any, update additional kwargs, but pl_module's opt has higher priority than additional kwargs
        for k,v in kwargs.items():
            if not k in nkwargs.keys():
                nkwargs[k] = v
        
        super().__init__(**nkwargs)
                
    def define_callbacks(self, opt):
        L = []
        if not opt.callbacks:
            return []
    
        callbacks = opt.callbacks.lower().split('_')
    
        # ModelCheckpoint
        if 'ckpt' in callbacks:
            cb_checkpoint = ModelCheckpoint(
                dirpath=os.path.join(opt.save_dir, 'checkpoint'),
                every_n_epochs=opt.checkpoint_every_n_epochs,
                monitor=opt.checkpoint_monitor,
                mode=opt.checkpoint_monitor_mode,
                filename=opt.checkpoint_filename,
                auto_insert_metric_name=False,
                save_weights_only=not opt.save_fullmodel,
                save_top_k=opt.checkpoint_save_top_k,
                #save_top_k=(-1 if opt.checkpoint_nooverwrite else 1),
            )
            L.append(cb_checkpoint)
            
        # LearingRateMonitor
        if 'lr' in callbacks:
            cb_lrmonitor = LearningRateMonitor(logging_interval='epoch')
            L.append(cb_lrmonitor)
            
        # Save Best Validation Metric
        if 'metricvalid' in callbacks:
            metric_tgt = []
            optmetrics = opt.metric.lower().split('_')
            if 'iou' in optmetrics:
                metric_tgt.append('metric/val_mIoU')
            if 'f1' in optmetrics:
                metric_tgt.append('metric/val_F1')
            
            cb_metricvalid = MetricsBestValidCallback(metric_tgt, opt.checkpoint_monitor, opt.checkpoint_monitor_mode)
            L.append(cb_metricvalid)
            
        # Results
        if 'result' in callbacks:
            cb_result = ResultsCallback(opt.result_dir, opt.result_save_npy,  opt.result_save_png, opt.postprocess)
            L.append(cb_result)

        # Simple Results
        if 'simpleresult' in callbacks:
            cb_result = SimpleResultsCallback(opt.result_dir, opt.result_save_npy,  opt.result_save_png, opt.postprocess)
            L.append(cb_result)
            
        # Bbox Results
        if 'bboxresult' in callbacks:
            cb_result = BboxResultsCallback(opt.result_dir)
            L.append(cb_result)
            
        return L
    
    def define_loggers(self, opt):
        save_dir_split = opt.save_dir.split(os.sep)
            
        save_dir_split = ['.','.'] + save_dir_split
        log_version = save_dir_split.pop()
        log_name = save_dir_split.pop()
        log_dir = os.path.join(*save_dir_split)
        
        L = []
        if (opt.loggers):
            loggers = opt.loggers.lower().split('_')
            
            if 'csv' in loggers:        # CSV Logger       
                L.append(CSVLogger(log_dir, name=log_name, version=log_version))
            if 'tb' in loggers:
                L.append(TensorBoardLogger(log_dir, name=log_name, version=log_version, sub_dir='tensorboard'))
            if 'wandb' in loggers:
                w_name = opt.wandb_name if opt.wandb_name else os.path.join(log_name, log_version)
                if '/' in opt.wandb_project:
                    entity = opt.wandb_project.split('/')[0]
                    project = opt.wandb_project.split('/')[1]
                    wlogger = WandbLogger(save_dir=opt.save_dir, project=project, entity=entity, name=w_name)
                else:
                    wlogger = WandbLogger(save_dir=opt.save_dir, project=opt.wandb_project, name=w_name)
                L.append(wlogger)
        
        if len(L)==0:
            L = False
        
        return L
