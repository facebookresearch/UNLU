#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

export MASTER_PORT=88888

echo $CUDA_VISIBLE_DEVICES
nvidia-smi
# End visible GPUs.

# setup conda environment
source setup.sh

which python

# python -m torch.distributed.launch --nproc_per_node=8
# CUDA_VISIBLE_DEVICES=1 python src/nli/training.py \
python src/nli/training.py \
  --model_class_name "roberta-base" \
  -n 1 \
  -g 1 \
  -nr 0 \
  --single_gpu \
  --fp16 \
  --fp16_opt_level O2 \
  --max_length 156 \
  --gradient_accumulation_steps 1 \
  --per_gpu_train_batch_size 64 \
  --per_gpu_eval_batch_size 16 \
  --save_prediction \
  --train_data \
  snli_train:none \
  --train_weights \
  1 \
  --eval_data \
  snli_dev:none \
  --eval_frequency 2000 \
  --train_with_lm \
  --lm_lambda 0.1 \
  --add_lm \
  --experiment_name "roberta-base|snli|lm_loss_0.1|add_lm"
#  \
# --learning_rate 0.001
# --train_with_lm \
# --lm_lambda 0.01 \
# --single_gpu \
