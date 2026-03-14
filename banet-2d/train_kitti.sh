#!/bin/bash

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0,1

# KITTI 微调训练脚本
# 请根据实际情况修改路径

# 预训练权重路径 (Scene Flow 预训练模型)
RESTORE_CKPT="./checkpoints/sceneflow/banet_sceneflow.pth"

# KITTI 数据集根目录
# 目录结构应包含 2012 和 2015 文件夹
KITTI_PATH="/data/StereoDatasets/kitti"

# 实验名称
EXP_NAME="banet_kitti"

python train_stereo.py \
    --name ${EXP_NAME} \
    --restore_ckpt ${RESTORE_CKPT} \
    --train_datasets kitti \
    --kitti_path ${KITTI_PATH} \
    --batch_size 4 \
    --lr 0.0004 \
    --num_steps 50000 \
    --image_size 320 1024 \
    --max_disp 192

# 注意:
# 1. KITTI 图像较大，建议减小 batch_size (如 4) 并增大 image_size (如 320x1024) 以覆盖更多区域
# 2. 学习率通常比从头训练要小
# 3. 确保 RESTORE_CKPT 指向正确的文件
