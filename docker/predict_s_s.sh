#!/bin/sh

GPU=0
DATAROOT="./inputs"
RUN_DIR="./outputs"
BATCH_SIZE=1
PATH_DS1="./parameters/coarse"
PATH_CK1="./parameters/coarse"
PATH_DS2="./parameters/fine"
PATH_CK2="./parameters/fine"
TEST_PREV_DIR="${RUN_DIR}/temp"

python -m mislight.scripts.test --no_saveoptions --gpu_ids $GPU --dataroot $DATAROOT --run_base_dir $RUN_DIR --batch_size $BATCH_SIZE --train_ds_info $PATH_DS1 --load_pretrained $PATH_CK1 --force_cpu_process --inference_model 2
python -m mislight.scripts.postprocess --coarse_export --no_cleanup --no_saveoptions --gpu_ids $GPU --dataroot $DATAROOT --run_base_dir $RUN_DIR --batch_size $BATCH_SIZE --train_ds_info $PATH_DS1 --load_pretrained $PATH_CK1 --force_cpu_process
python -m mislight.scripts.test --no_saveoptions --gpu_ids $GPU --dataroot $DATAROOT --run_base_dir $RUN_DIR --batch_size $BATCH_SIZE --train_ds_info $PATH_DS2 --load_pretrained $PATH_CK2 --dir_previous $TEST_PREV_DIR --force_cpu_process --inference_model 2
python -m mislight.scripts.postprocess --no_saveoptions --gpu_ids $GPU --dataroot $DATAROOT --run_base_dir $RUN_DIR --batch_size $BATCH_SIZE --train_ds_info $PATH_DS2 --load_pretrained $PATH_CK2 --force_cpu_process
