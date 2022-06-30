import copy
import shutil
from tqdm.autonotebook import tqdm

import torch

from mislight.options import TestOptions as Options
from mislight.data.datamodule import MyDataModule
from mislight.models import create_model, find_model_using_name
from mislight.engine.preprocessing import MyPreprocessing
from mislight.engine.postprocessing import MyPostprocessing
from mislight.engine.trainer import MyTrainer

if __name__ == '__main__':
    opt = Options().parse()
    
    ### options for datamodule & model
    mopt = copy.deepcopy(opt)
    mopt.datadir = opt.result_dir
    if opt.coarse_export:
        mopt.result_dir = opt.result_dir
    else:
        mopt.result_dir = opt.run_base_dir
    mopt.file_extension = 'npy'
    if opt.force_cpu_process:
        mopt.gpu_ids = None
                
    ### postprocess
    postp = MyPostprocessing(mopt)
    postp.run()
    
    ### cleanup
    if (not opt.coarse_export) and (not opt.no_cleanup):
        shutil.rmtree(opt.save_dir)
        print(f'cleanup {opt.save_dir}')
        shutil.rmtree(mopt.datadir)
        print(f'cleanup {mopt.datadir}')
