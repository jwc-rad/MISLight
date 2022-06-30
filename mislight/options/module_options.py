import argparse
import os
import json
import shutil

from mislight.utils.misc import make_dir_with_number, find_dir_with_number, find_file

from . import BaseOptions

class ModuleOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        # dataset
        parser.add_argument('--dataset_mode', type=str, default='segmentation', help='chooses how datasets are loaded')
        parser.add_argument('--dataset_mode_infer', type=str, default='segmentation_singletest', help='chooses how datasets are loaded for inference')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

        # model
        parser.add_argument('--load_pretrained', type=str, help="load pretrained weights only from checkpoint path")
        parser.add_argument('--checkpoint_query', type=str, default='', help="checkpoint path should contain this string")
        parser.add_argument('--single_precision', action='store_true', help='if True, fp32 instead of mixed precision')
        
        # log
        parser.add_argument('--log_type', type=str, help='CSV, TensorBoard. e.g., "csv_tb"')
        
        # inference
        parser.add_argument('--no_gaussian_weight', action='store_true', help='weight for crop, gaussian weight or 1')
        parser.add_argument('--no_largest_cc', action='store_true', help="if True, no largest connected component postprocessing")
        parser.add_argument('--coarse_export', action='store_true', help="if True, export binary mask numpy without resampling")
        parser.add_argument('--inference_model', type=float, default=0.5, help='which net to use at inference. depends on models')
        parser.add_argument('--force_cpu_process', action='store_true', help="force cpu for image interpolation in pre and post processing")
        
        
        return parser
