# Submission via Docker

## How Dockerfile looks like
```Dockerfile
FROM nvcr.io/nvidia/pytorch:21.10-py3
    
# Install some basic utilities and python
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Install requirements first
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY mislight ./mislight
COPY parameters ./parameters

COPY predict_s_s.sh ./predict.sh
```

## Prepare
First, copy module and other files into <code>dockerdir</code> (directory containing Dockerfile). 
```python
import os, distutils, shutil
import mislight

srcdir = os.path.dirname(mislight.__file__)
dstdir = os.path.join(dockerdir, 'mislight')

distutils.dir_util._path_created = {}
distutils.dir_util.copy_tree(srcdir, dstdir)

# delete ignored directories
del_dirs = []
for x in os.walk(dstdir):
    if '.git' in x[0]:
        continue
    if '__pycache__' in x[0]:
        del_dirs.append(x[0])
    if 'ipynb_checkpoints' in x[0]:
        del_dirs.append(x[0]) 
for x in del_dirs:
    shutil.rmtree(x)
    
# copy other files
srcfiles = [
    'docker/predict_s_s.sh',
    'docker/predict_s_t1.sh',
    'docker/predict_onestage_s.sh',
    'docker/predict_onestage_t1.sh',
    'docker/requirements.txt'
]
for x in srcfiles:
    shutil.copy(x, os.path.join(dockerdir, os.path.basename(x)))
    
```

Copy parameter files (pretrained model checkpoint and train_dataset_info json file) into <code>dockerdir</code>.
```python
parambase1 = 'COARSE_RUN_DIR'
stripdown1 = True # if using Teacher model for inference, set to False

parambase2 = 'FINE_RUN_DIR'
stripdown2 = True # if using Teacher model for inference, set to False

# COARSE
model_path = find_last_checkpoint(os.path.join(parambase1, 'checkpoint'))
ds_path = os.path.join(parambase1, 'dataset.json')

paramfiles = [model_path, ds_path]
newnames = ['model.ckpt', 'dataset.json']

dockerparamdir = os.path.join(dockerdir, 'parameters', 'coarse')
os.makedirs(dockerparamdir, exist_ok=True)

for x,n in zip(paramfiles, newnames):
    shutil.copy(x, os.path.join(dockerparamdir, n))
    
if stripdown1:
    new_model_path = os.path.join(dockerparamdir, 'model.ckpt')
    new_ckpt = torch.load(new_model_path, map_location='cpu')
    copy_state = new_ckpt['state_dict'].copy()
    for k in copy_state.keys():
        if not k.startswith('netS'):
            del new_ckpt['state_dict'][k]
    torch.save(new_ckpt, new_model_path)
    
shutil.copy(x, os.path.join(dockerdir, 'coarse_opt.txt'))
    
with open(os.path.join(dockerdir, 'coarse_parameters.json'), 'w') as f:
    json.dump(paramfiles, f)

# FINE
model_path = find_last_checkpoint(os.path.join(parambase2, 'checkpoint'))
ds_path = os.path.join(parambase2, 'dataset.json')

paramfiles = [model_path, ds_path]
newnames = ['model.ckpt', 'dataset.json']

dockerparamdir = os.path.join(dockerdir, 'parameters', 'fine')
os.makedirs(dockerparamdir, exist_ok=True)

for x,n in zip(paramfiles, newnames):
    shutil.copy(x, os.path.join(dockerparamdir, n))
    
if stripdown2:
    new_model_path = os.path.join(dockerparamdir, 'model.ckpt')
    new_ckpt = torch.load(new_model_path, map_location='cpu')
    copy_state = new_ckpt['state_dict'].copy()
    for k in copy_state.keys():
        if not k.startswith('netS'):
            del new_ckpt['state_dict'][k]
    torch.save(new_ckpt, new_model_path)
    
shutil.copy(x, os.path.join(dockerdir, 'fine_opt.txt'))
    
with open(os.path.join(dockerdir, 'fine_parameters.json'), 'w') as f:
    json.dump(paramfiles, f)
```

## Build
```bash
docker build -f {dockerdir}/Dockerfile -t {imagename}:latest {dockerdir}
```

## Run
```bash
docker container run --name {imagename} --gpus "device=0" --rm -v {LOCAL_INPUT_DIR}:/workspace/inputs/ -v {LOCAL_OUTPUT_DIR}:/workspace/outputs/ {imagename}:latest /bin/bash -c "sh predict.sh"
```
or for FLARE22 evaluation
```bash
cd flare_eval
nohup python local_resource_eval.py --gpu_id 0 --docker_name {imagename} >> infos.log &
```
