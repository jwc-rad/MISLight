#!/bin/sh

GPU=0
DATAROOT="./inputs"
RUN_DIR="./outputs"
BATCH_SIZE=1
PATH_DS="./parameters"
PATH_CK="./parameters"

python -m mislight.scripts.test --no_saveoptions --gpu_ids $GPU --dataroot $DATAROOT --run_base_dir $RUN_DIR --batch_size $BATCH_SIZE --train_ds_info $PATH_DS --load_pretrained $PATH_CK --force_cpu_process --inference_model 0
python -m mislight.scripts.postprocess --no_saveoptions --gpu_ids $GPU --dataroot $DATAROOT --run_base_dir $RUN_DIR --batch_size $BATCH_SIZE --train_ds_info $PATH_DS --load_pretrained $PATH_CK --force_cpu_process
