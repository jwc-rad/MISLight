<div align="center">    
 
# MISLight
Medical Image Segmentation in Pytorch Lightning
</div>

## Description
**MISLight** is just one of many medical image segmentation implementations out there, yet it aims for readability and reusability in line with the philosophy of [Pytorch Lightning](https://github.com/Lightning-AI/lightning).
- This [branch](https://github.com/jwc-rad/MISLight/tree/flare22) is the official repository for **Knowledge Distillation from Cross Teaching Teachers for Efficient Semi-supervised Abdominal Organ Segmentation in CT**, which was used for the [MICCAI FLARE 2022](https://flare22.grand-challenge.org) challenge.

## Usage
### Dataset format (follows [nnU-Net](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md))
```
TRAIN_DATASET_DIR/
├── TrainImage
│   ├── LABELED_0001_0000.nii.gz
│   ├── LABELED_0002_0000.nii.gz
│   ├── ...
│   ├── NOLABEL_0001_0000.nii.gz
│   ├── NOLABEL_0002_0000.nii.gz
│   ├── ...
├── TrainMask
│   ├── LABELED_0001.nii.gz
│   ├── LABELED_0002.nii.gz
│   ├── ...
```

### Preprocessing for whole-volume-based coarse segmentation
```bash
python -m mislight.scripts.preprocess --gpu_ids -1 --run_base_dir $COARSE_PREPROCESS_DIR --dataroot $TRAIN_DATASET_DIR --resample_target size --resample_fixed 96 96 96 --ipl_order_image 1 --ipl_order_mask 1
```
### Train coarse model
```bash
python -m mislight.scripts.train --force_cpu_process --gpu_ids 0 --dataroot $COARSE_PREPROCESS_DIR --dir_image TrainImage --dataset_mode segmentation_ssl --crop_size 96 64 96 --windowHU 600 0 --batch_size 1 --model cross_teach_kd --netS mobile_se_resunet se_resunet se_resunet --n_blocks_per_stage 1 --nsf 32 32 32 --n_stages 5 --transposed_conv 1 1 0 --log_type tb --checkpoint_every_n_epochs 250 --data_augmentation v4 --inference_model 2 --exp_name coarse --exp_number 1 --n_epochs 1000 --fold -1
```
### Coarse segmentation
This yields coarse segmentation masks for all data including those with ground truth masks under <code>$FINE_PREPROCESS_DIR/previous/temp</code>. Before preprocessing for fine segmentation, copy the coarse masks for unlabeled data and ground truth masks for labeled data to <code>$FINE_PREPROCESS_DIR/previous</code>.
```bash
python -m mislight.scripts.test_nopreprocess --force_cpu_process --gpu_ids 0 --dataroot $COARSE_PREPROCESS_DIR --dir_image TrainImage --batch_size 1 --train_ds_info ./runs/coarse/00001 --load_pretrained ./runs/coarse/00001/checkpoint --inference_model 2 --run_base_dir $FINE_PREPROCESS_DIR/previous
```
### Preprocessing for fine segmentation
```bash
python -m mislight.scripts.preprocess --gpu_ids -1 --run_base_dir $FINE_PREPROCESS_DIR --dataroot $TRAIN_DATASET_DIR --resample_target size --resample_fixed 96 64 96 --ipl_order_image 1 --ipl_order_mask 1 --dir_previous $FINE_PREPROCESS_DIR/previous
```
### Train fine model
```bash
python -m mislight.scripts.train --force_cpu_process --gpu_ids 0 --dataroot $FINE_PREPROCESS_DIR --dataset_mode segmentation_ssl --crop_size 96 64 96 --windowHU 600 0 --batch_size 1 --model cross_teach_kd --netS mobile_se_resunet se_resunet se_resunet --n_blocks_per_stage 1 --nsf 32 32 32 --n_stages 5 --transposed_conv 1 1 0 --log_type tb --checkpoint_every_n_epochs 250 --data_augmentation v4 --inference_model 2 --exp_name fine --exp_number 1 --n_epochs 1000 --fold -1
```
### Test and post-processing
For inference on test set, see [test script](https://github.com/jwc-rad/MISLight/blob/flare22/docker/predict_s_s.sh) and [docker instructions](https://github.com/jwc-rad/MISLight/tree/flare22/docker).

## Acknowledgement
- This project includes codes from the following projects: [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [SSL4MIS](https://github.com/HiLab-git/SSL4MIS), [MONAI](https://github.com/Project-MONAI/MONAI), [Loss Odyssey](https://github.com/JunMa11/SegLoss)
