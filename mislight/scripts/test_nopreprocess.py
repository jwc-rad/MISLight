import copy
import os
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
    
    ### opt.datadir should be already preprocessed datadir
    
    ### options for datamodule & model
    mopt = copy.deepcopy(opt)
    shutil.copy(os.path.join(mopt.dataroot, 'dataset.json'), mopt.dataset_json)
    mopt.file_extension = 'npy'

    # update from trained model, but specificed options can be overriden
    ckpt = torch.load(opt.load_pretrained, map_location=torch.device('cpu'))
    old_opt = ckpt['hyper_parameters']['opt']
    del ckpt
    mopt.model = old_opt.model
    model_class = find_model_using_name(mopt.model)
    model_args = model_class.get_model_specific_args()
    for k,v in old_opt.__dict__.items():
        if k in model_args:
            mopt.__dict__[k] = v
    
    ### run test
    dm = MyDataModule(mopt)
    model = create_model(mopt)
    model.load_pretrained(mopt.load_pretrained)
    trainer = MyTrainer(dm)
    trainer.test(model, dm, verbose=False)
