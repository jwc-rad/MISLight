import argparse
import collections.abc
import os
import yaml
import torch

from mislight.data import find_dataset_using_name
from mislight.models import find_model_using_name
from mislight.utils.misc import make_dir_with_number, find_dir_with_number, find_file

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.phase = None
        
    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders TrainImage, etc.)')
        parser.add_argument('--dir_image', type=str, help='path to images under dataroot')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--run_base_dir', type=str, default='./runs', help='models are saved here (train) or loaded from here (test)')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--file_extension', type=str, default='npy', help='case-sensitive image file extension')
        parser.add_argument('--no_saveoptions', action='store_true')
        
        parser.add_argument('--ipl_order_image', default=1, type=int, help='interpolation order for image')
        parser.add_argument('--ipl_order_mask', default=1, type=int, help='interpolation order for mask')
        
        # for coarse to fine
        parser.add_argument('--dir_previous', type=str, help='path to previous masks (for coarse to fine refinement). should contain numpy arrays.')
        parser.add_argument('--padding_buffer', type=float, default=1.1, help='buffer padding in previous mask')
        parser.add_argument('--input_min_size', type=float, nargs=3, default=[240,160,240], help='minimum input size in mm.')
                
        self.initialized = True
        return parser

    def gather_options(self, args=None):
        '''
        Add additional model-specific and dataset-specific options.
        Parameters:
            args: only for using in jupyter notebook
        '''
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            
        # get the basic options
        if args is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(args=args)
        
        if 'model' in opt:
            # modify model-related parser options
            model_name = opt.model
            model_class = find_model_using_name(model_name)
            parser = model_class.add_model_specific_args(parser)
            if args is None:
                opt, _ = parser.parse_known_args()
            else:
                opt, _ = parser.parse_known_args(args=args)

        if 'dataset_mode' in opt:
            # modify dataset-related parser options
            ds_name = opt.dataset_mode
            ds_class = find_dataset_using_name(ds_name)
            parser = ds_class.add_dataset_specific_args(parser)
            if args is None:
                opt, _ = parser.parse_known_args()
            else:
                opt, _ = parser.parse_known_args(args=args)
                
        if 'dataset_mode_infer' in opt:
            # modify dataset-related parser options
            ds_name = opt.dataset_mode_infer
            ds_class = find_dataset_using_name(ds_name)
            parser = ds_class.add_dataset_specific_args(parser)
            if args is None:
                opt, _ = parser.parse_known_args()
            else:
                opt, _ = parser.parse_known_args(args=args)
        
        if args is None:
            opt = parser.parse_args()
        else:
            opt = parser.parse_args(args=args) 
        
        # save and return the parser
        self.parser = parser
        return opt
    
    def parse(self, args=None):
        opt = self.gather_options(args=args)
        opt.phase = self.phase
        opt.isTrain = (opt.phase == 'train')
        opt = self.setup(opt)
        opt = self.print_options(opt)
            
        # parse gpu ids as used in Pytorch Lightning Trainer 
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) == 0:
            opt.gpu_ids = None # CPU
            
        self.opt = opt
        return self.opt
    
    def setup(self, opt):
        '''Should be defined in child classes
        '''
        return opt
    
    def print_options(self, opt):
        '''Print options and save
        '''
        
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in vars(opt).items():
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)        
        
        if not opt.no_saveoptions:
            file_name = os.path.join(opt.save_dir, f'_{opt.phase}_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')        
            print(f'saved {file_name}')

            '''
            yaml_name = os.path.join(opt.save_dir, f'_{opt.phase}_opt.yaml')
            with open(yaml_name, 'w') as f:
                yaml.dump(vars(opt), f)
            print(f'saved {yaml_name}')
            '''
        
        return opt
