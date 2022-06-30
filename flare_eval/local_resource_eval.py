'''Evaluation script for local validation set. Docker image is expected to be already loaded. Only call it by docker_name:latest.
'''
import argparse
import glob
import os
import shutil
import time
import torch
from pathlib import Path
join = os.path.join
from logger import add_file_handler_to_logger, logger

add_file_handler_to_logger(name="main", dir_path="logs/", level="DEBUG")

def check_dir(file_path):
    file_path = Path(file_path)
    files = [f for f in file_path.iterdir() if ".nii.gz" in str(f)]
    if len(files) != 0:
        return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker_name", required=True, help="image must be loaded first")
    parser.add_argument("--test_img_path", default='../temp/FLARE22_valid/')
    parser.add_argument("--save_path", default='../results/docker/')
    parser.add_argument("--gpu_id", default=1)
    args = parser.parse_args()
    
    docker = args.docker_name
    test_img_path = args.test_img_path
    save_path = args.save_path
    gpu_id = args.gpu_id
    
    name = docker.split('.')[0].lower()
    team_outpath = join(save_path, name)
    os.makedirs(team_outpath, exist_ok=True)
    
    test_cases = sorted(os.listdir(test_img_path))
    os.makedirs('./inputs/', exist_ok=True)
    os.makedirs('./outputs/', exist_ok=True)
    
    try:
        #name = docker.split('.')[0].lower()
        print('teamname docker: ', docker)
        #os.system('docker image load < {}'.format(join(docker_path, docker)))
        #team_outpath = join(save_path, name)
        if os.path.exists(team_outpath):
            shutil.rmtree(team_outpath)
        os.mkdir(team_outpath)
        for case in test_cases:
            if not check_dir('./inputs'):
                logger.error("please check inputs folder")
                raise
            shutil.copy(join(test_img_path, case), './inputs')
            start_time = time.time()
            os.system('python Efficiency.py -gpus {} -docker_name {} -save_file {}'.format(gpu_id, name, save_path))
            logger.info(f"{case} finished!")
            os.remove(join('./inputs', case))
            # shutil.rmtree('./inputs')
            logger.info(f"{case} cost time: {time.time() - start_time}")

        os.system("python load_json.py -docker_name {} -save_path {}".format(name, save_path))
        shutil.move("./outputs", team_outpath)
        os.mkdir("./outputs")
        torch.cuda.empty_cache()
        # os.system("docker rmi {}:latest".format(name))
    except Exception as e:
        logger.exception(e)
