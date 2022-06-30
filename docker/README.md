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

COPY predict.sh .
```

## Prepare
First, copy module into <code>dockerdir</code> (directory containing Dockerfile). 
```python
import os
import distutils

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
```

Copy parameter files (pretrained model checkpoint and train_dataset_info json file) into <code>dockerdir</code>.
```python
paramfiles = [
    'SOMEWHERE/pretrained.ckpt',
    'SOMEWHERE/dataset.json',
]

dockerparamdir = os.path.join(dockerdir, 'parameters')
os.makedirs(dockerparamdir, exist_ok=True)
for x in paramfiles:
    shutil.copy(x, os.path.join(dockerparamdir, os.path.basename(x)))
```

## Build
```bash
docker build -f {dockerdir}/Dockerfile -t {imagename}:{tagname} {dockerdir}
```

## Run
```bash
docker container run --name {imagename} --gpus "device=0" --rm -v {LOCAL_INPUT_DIR}:/workspace/inputs/ -v {LOCAL_OUTPUT_DIR}:/workspace/outputs/ {imagename}:{tagname} /bin/bash -c "sh predict.sh"
```
