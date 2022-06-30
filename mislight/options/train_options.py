import argparse
import os
import json
import shutil

from mislight.utils.misc import make_dir_with_number, find_dir_with_number, find_last_checkpoint

from . import ModuleOptions

class TrainOptions(ModuleOptions):
    def initialize(self, parser):
        parser = ModuleOptions.initialize(self, parser)
        
        # experiment
        parser.add_argument('--exp_name', type=str, default='exp', help='experiment name')
        parser.add_argument('--exp_number', type=int, default=0, help='if 0, make new folder under run_base_dir/exp_name. if >0, find existing folder or make new folder with the number. if folder exists, suffix is ignored.')
        parser.add_argument('--exp_suffix', type=str, default='', help='Becomes suffix of save_dir. e.g., run_base_dir/{exp_name}/{exp_number}_{exp_suffix}')
        parser.add_argument('--fold', type=str, default='-1', help='-1 for no fold. if use, e.g. 0,1,2 in 5-fold')
        
        # model
        parser.add_argument('--model', type=str, default='fsl', help='chooses which model to use.')    

        # training parameters
        parser.add_argument('--train_only', action='store_true', help='no run validation inference')
        parser.add_argument('--resume_from_checkpoint', type=str, help="checkpoint path. if use, this will override load_pretrained. Used in Trainer's resume_from_checkpoint.")
        parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs')
        parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate for SGD. nnUNet default')
        parser.add_argument('--weight_decay', type=float, default=3e-5, help='decay term of SGD. nnUNet default')
        parser.add_argument('--momentum', type=float, default=0.99, help='momentum term of SGD. nnUNet default')
        parser.add_argument('--no_nesterov', action='store_true', help='if True, nesterov is False for SGD')
        parser.add_argument('--lr_policy', type=str, default='poly', help='learning rate policy. nnUNet default is poly [poly | none | linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='not used. multiply by a gamma every lr_decay_iters iterations, if lr_policy==step')
        parser.add_argument('--save_fullmodel', action='store_true', help='if True, save full model. Else, save weights only')
        parser.add_argument('--data_augmentation', type=str, default='v4', help='data augmentation versions')
        
        # trainer
        parser.add_argument('--checkpoint_nooverwrite', action='store_true', help='keep every saved checkpoint. passed to ModelCheckpoint')
        parser.add_argument('--checkpoint_every_n_epochs', type=int, help='checkpoint every n epochs. passed to ModelCheckpoint')
        parser.add_argument('--checkpoint_every_n_train_steps', type=int, help='checkpoint every n steps. passed to ModelCheckpoint')
        parser.add_argument('--checkpoint_filename', type=str, default='{epoch}_{step}', help='checkpoint file format')
        parser.add_argument('--log_every_n_steps', type=int, default=10, help='logging frequency')
        parser.add_argument('--detect_anomaly', action='store_true', help='detect anomaly')
        
        # validation inference
        parser.add_argument('--eval_only', action='store_true', help='run validation inference only')
        parser.add_argument('--export_metrics', action='store_true', help='export FLARE metrics for validation')
                
        self.phase = 'train'
        return parser
    
    def setup(self, opt):
        opt.inference = False
        ### total epochs should be multiple of checkpoint save frequency
        assert (opt.n_epochs / opt.checkpoint_every_n_epochs).is_integer()
        
        ### handle batch norm
        if opt.norm == 'bn':
            assert opt.batch_size > 1
        if (opt.batch_size > 1) and (opt.norm == 'bn'):
            opt.batch_drop_last = True
        
        ### resume_from_checkpoint
        if opt.resume_from_checkpoint:
            if os.path.exists(opt.resume_from_checkpoint):
                if os.path.isdir(opt.resume_from_checkpoint):
                    load_dir = opt.resume_from_checkpoint
                    load_path = find_last_checkpoint(load_dir, query=opt.checkpoint_query)
                    opt.resume_from_checkpoint = load_path
            else:
                print(f'cannot find {opt.resume_from_checkpoint}')
                opt.resume_from_checkpoint = None
                
        ### save dir
        run_dir = os.path.join(opt.run_base_dir, opt.exp_name)
        if opt.exp_number==0:
            save_dir = make_dir_with_number(run_dir, fname=opt.exp_suffix)
        else:
            try:
                save_dir = find_dir_with_number(opt.exp_number, run_dir)
                if save_dir is None:
                    save_dir = make_dir_with_number(run_dir, fname=opt.exp_suffix, num=opt.exp_number)
            except:
                save_dir = make_dir_with_number(run_dir, fname=opt.exp_suffix, num=opt.exp_number)    
        opt.save_dir = save_dir
        opt.result_dir = None
        
        ### datadir
        if opt.dir_image:
            opt.datadir = os.path.join(opt.dataroot, opt.dir_image)
        else:
            opt.datadir = opt.dataroot
        
        ### dataset
        opt.dataset_json = os.path.join(opt.save_dir, 'dataset.json')
        shutil.copy(os.path.join(opt.dataroot, 'dataset.json'), opt.dataset_json)       
        with open(opt.dataset_json, 'r') as f:
            ds_json = json.load(f) 
        update_parameters = ['ipl_order_image', 'ipl_order_mask', 'num_classes', 'nc_input']
        for k in update_parameters:
            opt.__dict__[k] = ds_json[k]
        opt.dataset_split = os.path.join(opt.save_dir, 'split.pkl')

        str_folds = opt.fold.split(',')
        opt.fold = []
        for str_fold in str_folds:
            fold = int(str_fold)
            if fold >= 0:
                opt.fold.append(fold)
        if len(opt.fold) == 0:
            opt.fold = [-1] # no fold
        
        return opt
