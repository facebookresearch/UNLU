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
export BUILD_LOC=/private/home/koustuvs/mlp/nli_gen/anli/data/build/mnli
# python -m torch.distributed.launch --nproc_per_node=8 src/nli/training.py \
python src/nli/training.py \
  --model_class_name "roberta-base" \
  -n 1 \
  -g 8 \
  --fp16 \
  --fp16_opt_level O2 \
  -nr 0 \
  --max_length 156 \
  --gradient_accumulation_steps 4 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 16 \
  --save_prediction \
  --train_data \
  mnli_rand_0.25:"${BUILD_LOC}/rand_p_0.25_train.jsonl",mnli_rand_0.5:"${BUILD_LOC}/rand_p_0.5_train.jsonl",mnli_rand_0.75:"${BUILD_LOC}/rand_p_0.75_train.jsonl",mnli_rand_1.0:"${BUILD_LOC}/rand_p_1.0_train.jsonl" \
  --train_weights \
  1,1,1,1 \
  --eval_data \
  mnli_m_dev:none \
  --eval_frequency 2000 \
  --experiment_name "roberta-base|mnli_nli|rand_all_train"
