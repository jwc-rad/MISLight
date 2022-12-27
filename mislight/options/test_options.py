import argparse
import collections.abc
import os
import yaml
import torch

from mislight.utils.misc import make_dir_with_number, find_dir_with_number, str2bool

class TestOptions():
    def __init__(self):
        self.initialized = False
        
    def initialize(self, parser):
        # basic
        parser.add_argument('--run_base_dir', type=str, default='./results', help='models are saved here (train) or loaded from here (test)')
        parser.add_argument('--fold', type=int, nargs='+', help='-1 for no fold. if use, e.g. 0 1 2 in 5-fold')
        parser.add_argument('--gpu_ids', type=int, default=[0], nargs='+', help='gpu ids: e.g. 0 1. use -1 for CPU')
        parser.add_argument('--num_workers', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--saveoptions', default=True, type=str2bool)
        parser.add_argument('--mixed_precision', default=False, type=str2bool, help='if True, use AMP')
        parser.add_argument('--save_temp', default=False, type=str2bool, help='only used in ensemble. if True, save individual results')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        # data
        parser.add_argument('--phase', type=str, help='name of dataset class')
        parser.add_argument('--dataset_mode', type=str, help='name of dataset class')
        parser.add_argument('--datadir', type=str, help='path to data')
        parser.add_argument('--maskdir', type=str, help='optional')
        parser.add_argument('--dataset_split', type=str, help='path to dataset split.pkl')
        parser.add_argument('--num_class', type=int, default=2, help='number of classes including background')
        parser.add_argument('--load_size', default=512, type=int, help='resize loaded image')
        #parser.add_argument('--load_max_size', type=int, help='max for resize loaded image')
        parser.add_argument('--load_mode', default='shortest', type=str, help='refer to ResizeV2 size_mode')
        parser.add_argument('--patch_size', default=256, type=int, help='input patch size')
        #parser.add_argument('--patch_resize_factor', default=1, type=float, help='resize patch. 2 means half')
        parser.add_argument('--patch_overlap_inference', default=0.5, type=float, help='overlap for sliding window inference')
        parser.add_argument('--batch_size', default=1, type=int, help='batch size for sliding window')
        parser.add_argument('--batch_size_inference', default=1, type=int, help='batch size for inference')
        
        # model
        parser.add_argument('--load_pretrained_model', type=str, nargs='+', help='path to pretrained model')
        parser.add_argument('--metric', type=str, default='f1_iou', help='iou, f1')
        parser.add_argument('--checkpoint_monitor', type=str, default='val_mIoU=', help='checkpoint monitor to look for in load_pretrained_model')
        parser.add_argument('--checkpoint_monitor_mode', type=str, default='min', help='e.g. min for losses. max for accuracy')
        
        # trainer
        parser.add_argument('--callbacks', type=str, help='result')
        parser.add_argument('--loggers', type=str, help='csv, tb, wandb')
        parser.add_argument('--wandb_project', type=str,)
        parser.add_argument('--wandb_name', type=str)
        parser.add_argument('--log_every_n_steps', type=int, default=10, help='logging frequency')
        
        # result parameters
        parser.add_argument('--result_save_npy', default=False, type=str2bool)
        parser.add_argument('--result_save_png', default=False, type=str2bool)
        parser.add_argument('--postprocess', default=False, type=str2bool)
            
        self.initialized = True
        return parser
        
    def parse(self, args=None):
        opt = self.gather_options(args=args)
        opt = self.setup(opt)
        opt = self.print_options(opt)
        
        self.opt = opt
        return opt
        
    def gather_options(self, args=None):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            
        if args is None:
            opt = parser.parse_args()
        else:
            opt = parser.parse_args(args=args)        
            
        # save and return the parser
        self.parser = parser
        return opt

    def setup(self, opt):
        opt.save_results = True
        ### result dir
        if opt.save_results:
            opt.result_dir = opt.save_dir = opt.run_base_dir 
            os.makedirs(opt.result_dir, exist_ok=True)
        
        opt.inference = True
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
        
        if opt.saveoptions:
            file_name = os.path.join(opt.save_dir, f'_test_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')        
            print(f'saved {file_name}')
        
        return opt