import copy
import os
import shutil
from tqdm.autonotebook import tqdm

from mislight.options import TrainOptions as Options
from mislight.data.datamodule import MyDataModule
from mislight.models import create_model
from mislight.engine.postprocessing import MyPostprocessing
from mislight.engine.trainer import MyTrainer
from mislight.utils.evaluation import export_metrics, CV_metrics
from mislight.utils.misc import find_last_checkpoint

if __name__ == '__main__':
    opt = Options().parse()
    
    for fold in opt.fold:
        if not opt.eval_only:
            newopt = copy.deepcopy(opt)
            newopt.fold = fold

            dm = MyDataModule(newopt)
            model = create_model(newopt)
            if opt.resume_from_checkpoint:
                trainer = MyTrainer(dm, resume_from_checkpoint=opt.resume_from_checkpoint)
            else:
                trainer = MyTrainer(dm)
            trainer.fit(model, dm)
        
        # infer
        if (not opt.train_only) and (fold >= 0):
            newopt = copy.deepcopy(opt)
            newopt.fold = fold
            newopt.inference = True
            newopt.log_type = None
            newopt.batch_size = 1 # for lower memory usage

            dm = MyDataModule(newopt)
            model = create_model(newopt)
            
            # get pretrained path
            if newopt.load_pretrained is None:
                load_dir = os.path.join(newopt.save_dir, 'checkpoint')
                load_path = find_last_checkpoint(load_dir, query=newopt.checkpoint_query)
                newopt.load_pretrained = load_path                
            elif os.path.isdir(newopt.load_pretrained):
                load_dir = newopt.load_pretrained
                load_path = find_last_checkpoint(load_dir, query=newopt.checkpoint_query)
                newopt.load_pretrained = load_path
            else:
                load_path = newopt.load_pretrained            
            assert os.path.exists(load_path)
            # get result dir (temp)
            if len(newopt.checkpoint_query) > 0:
                result_dir = os.path.join(newopt.save_dir, 'results_'+newopt.checkpoint_query, 'temp')
            else:
                result_dir = os.path.join(newopt.save_dir, 'results', 'temp')
            os.makedirs(result_dir, exist_ok=True)
            newopt.result_dir = result_dir

            # run inference
            model.load_pretrained(newopt.load_pretrained)
            trainer = MyTrainer(dm)
            trainer.test(model, dm, verbose=False)
            
            # postprocessing
            popt = copy.deepcopy(newopt)
            if newopt.force_cpu_process:
                popt.gpu_ids = None
            popt.datadir = newopt.result_dir
            popt.result_dir = os.path.dirname(newopt.result_dir)
            postp = MyPostprocessing(popt)
            postp.run()
            
            # cleanup
            shutil.rmtree(popt.datadir)
            print(f'cleanup {popt.datadir}')
            
            if opt.export_metrics:
                metrics_path = export_metrics(popt.result_dir, dm.ds['srcdirlabel'])
                print(f'saved metrics to {metrics_path}')
                
                
    if (not opt.train_only) and (opt.export_metrics):
        cv_metrics_path = CV_metrics(opt.save_dir)
        if cv_metrics_path:
            print(f'saved CV metrics to {cv_metrics_path}')
