import argparse
import os
import json
import shutil

from mislight.utils.misc import make_dir_with_number, find_dir_with_number, find_last_checkpoint

from . import ModuleOptions

class TestOptions(ModuleOptions):
    def initialize(self, parser):
        parser = ModuleOptions.initialize(self, parser)
        
        parser.set_defaults(run_base_dir='./results', file_extension='nii.gz', batch_size=1, dir_image=None)
        parser.add_argument('--train_ds_info', type=str, default='.', help="path to trained dataset info. contains preprocessing parameters, etc.")
        parser.add_argument('--no_cleanup', action='store_true', help='if True, no delete temporary folders')
                
        self.phase = 'test'
        return parser
    
    def setup(self, opt):
        ### load_pretrained
        if os.path.isdir(opt.load_pretrained):
            load_dir = opt.load_pretrained
            load_path = find_last_checkpoint(load_dir, query=opt.checkpoint_query)
            opt.load_pretrained = load_path
        assert opt.load_pretrained, 'no load_pretrained'
        assert os.path.exists(opt.load_pretrained), '{opt.load_pretrained} does not exist)'            
        
        ### dirs
        # save dir for preprocess
        opt.save_dir = os.path.join(opt.run_base_dir, 'preprocess')
        os.makedirs(opt.save_dir, exist_ok=True)
        # temp result dir for saving results
        opt.result_dir = os.path.join(opt.run_base_dir, 'temp')
        os.makedirs(opt.result_dir, exist_ok=True)
        
        ### datadir
        if opt.dir_image:
            opt.datadir = os.path.join(opt.dataroot, opt.dir_image)
        else:
            opt.datadir = opt.dataroot
        
        ### train_ds_info path
        if os.path.isdir(opt.train_ds_info):
            load_dir = opt.train_ds_info
            load_path = os.path.join(load_dir, 'dataset.json')
            opt.train_ds_info = load_path            
        assert os.path.exists(opt.train_ds_info), '{opt.train_ds_info} does not exist'
        with open(opt.train_ds_info, 'r') as f:
            train_ds = json.load(f)
        
        # update from train_ds_info
        update_parameters = ['resample_method', 'resample_target', 'ipl_order_image', 'ipl_order_mask', 'num_classes', 'nc_input']
        for k in update_parameters:
            opt.__dict__[k] = train_ds[k]
        
        # other updates
        opt.resample_fixed = train_ds[f'target_{opt.resample_target}']
        opt.dataset_json = os.path.join(opt.save_dir, 'dataset.json')
        opt.fold = -1
        
        opt.inference = True
                
        return opt
