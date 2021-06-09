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

CUDA_VISIBLE_DEVICES=1 python src/nli/train_with_scramble.py \
  --model_class_name "roberta-base" \
  -n 1 \
  -g 1 \
  --single_gpu \
  --fp16 \
  --fp16_opt_level O2 \
  -nr 0 \
  --max_length 156 \
  --gradient_accumulation_steps 4 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 16 \
  --save_prediction \
  --train_data \
  mnli_train:none \
  --train_weights \
  1 \
  --scrambled_train_data \
  mnli_rand_train:none \
  --scrambled_train_weights \
  1 \
  --eval_data \
  mnli_m_dev:none \
  --eval_frequency 2000 \
  --train_with_lm \
  --lm_lambda 0.1 \
  --entropy_lambda 0.5 \
  --add_lm \
  --experiment_name "roberta-base|mnli|scramble|train_with_entropy|"
