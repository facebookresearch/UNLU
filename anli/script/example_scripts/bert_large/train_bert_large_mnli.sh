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

python src/nli/training.py \
    --model_class_name "bert-large" \
    -n 1 \
    -g 8 \
    --fp16 \
    --fp16_opt_level O2 \
    -nr 0 \
    --max_length 156 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --save_prediction \
    --train_data \
    mnli_train:none \
    --train_weights \
    1 \
    --eval_data \
    mnli_m_dev:none \
    --eval_frequency 2000 \
    --experiment_name "bert-large|mnli_nli|b8"
