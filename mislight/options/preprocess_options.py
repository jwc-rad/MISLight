import argparse
import os

from mislight.utils.misc import make_dir_with_number, find_dir_with_number, find_file

from . import BaseOptions

class PreprocessOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)        
        parser.set_defaults(run_base_dir='./preprocess', file_extension='nii.gz')
        
        parser.add_argument('--inference', action='store_true', help='images only')
        parser.add_argument('--dir_label', type=str, default='TrainMask', help='path to images under dataroot')
        parser.add_argument('--resample_method', type=str, default='torch', help='torch or skimage or scipy')
        parser.add_argument('--resample_target', type=str, default='spacing', help='spacing or size')
        parser.add_argument('--resample_fixed', type=float, nargs=3, help='if used, fixed target')
        parser.add_argument('--check_resample_target', action='store_true')
        
        self.phase = 'preprocess'
        return parser

    def setup(self, opt):        
        opt.save_dir = opt.run_base_dir
        os.makedirs(opt.save_dir, exist_ok=True)
        
        ### datadir
        if opt.dir_image:
            opt.datadir = os.path.join(opt.dataroot, opt.dir_image)
        else:
            opt.datadir = opt.dataroot
        
        return 
