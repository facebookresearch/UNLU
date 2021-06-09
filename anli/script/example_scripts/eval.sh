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

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data snli \
#   --train_mode orig \
#   --eval_data snli_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data snli \
#   --train_mode orig \
#   --eval_data snli_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_m_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_mm_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data mnli_m_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data mnli_mm_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data mnli_m_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data mnli_mm_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data snli_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data snli_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data snli_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data anli_r1_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data anli_r2_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data anli_r3_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data anli_r1_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data anli_r2_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data anli_r3_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data anli_r1_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data anli_r2_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data anli_r3_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data sem_neg

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data sem_neg

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_neg

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data sem_quant

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data sem_quant

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_quant

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data sem_bool

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data sem_bool

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_bool

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data sem_count

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data sem_count

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_count

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data sem_cond

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data sem_cond

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_cond

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data sem_comp

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data sem_comp

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_comp

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data bnli

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data bnli

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data bnli

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data superglue_diag

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data superglue_diag

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data superglue_diag

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_rand_p_1.0

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_aug \
#   --eval_data mnli_rand_p_1.0

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data mnli_rand_p_1.0

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_m_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data mnli_mm_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data snli_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data snli_dev

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data anli_r1_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data anli_r2_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data anli_r3_test

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data mnli_rand_p_0.25

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data mnli_rand_p_0.5

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data mnli_rand_p_0.75

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data mnli_rand_p_1.0

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_neg

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_quant

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_bool

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_count

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_cond

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data sem_comp

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data hans

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data bnli

# python src/nli/evaluation.py \
#   --model_class_name "roberta-large" \
#   --train_data mnli \
#   --train_mode rand_entropy \
#   --eval_data superglue_diag

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_rand_cont_0

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_rand_cont_0.25

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_rand_cont_0.5

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_rand_cont_0.75

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_rand_cont_1.0

# python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_m_dev_rand_cont_0_seed

# CUDA_VISIBLE_DEVICES=0 python src/nli/evaluation.py \
#   --model_class_name "roberta-base" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data anli_r3_test_rand_cont_0_seed_all

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_m_dev

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_m_dev_rand_final \
#   --slurm

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data mnli_mm_dev_rand_final \
#   --slurm

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data snli_dev_rand_final \
#   --slurm

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data snli_test_rand_final \
#   --slurm

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data anli_r1_dev_rand_final \
#   --slurm

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data anli_r2_dev_rand_final \
#   --slurm

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data anli_r3_dev_rand_final \
#   --slurm

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data snli_dev

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data snli_test

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data anli_r1_dev

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "distilbert" \
#   --train_data mnli \
#   --train_mode orig \
#   --eval_data anli_r2_dev

# MODEL_CLASS_NAME="distilbert"
# TRAIN_DATA="mnli"
# TRAIN_MODE="orig"
# EVAL_DATA="mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev"

# for eval_data in $EVAL_DATA; do
#   CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#     --model_class_name $MODEL_CLASS_NAME \
#     --train_data $TRAIN_DATA \
#     --train_mode $TRAIN_MODE \
#     --eval_data "${eval_data}_rand_keep_premise" \
#     --slurm
# done

# CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#   --model_class_name "chinese-roberta-large" \
#   --train_data ocnli \
#   --train_mode orig \
#   --eval_data ocnli_dev_rand

# MODEL_CLASS_NAME="roberta-large"
# TRAIN_DATA="mnli"
# TRAIN_MODE="rand_final"
# EVAL_DATA="mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev"

# for eval_data in $EVAL_DATA; do
#   CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#     --model_class_name $MODEL_CLASS_NAME \
#     --train_data $TRAIN_DATA \
#     --train_mode $TRAIN_MODE \
#     --eval_data "${eval_data}"
# done

# MODEL_CLASS_NAME="bart-large"
# TRAIN_DATA="mnli"
# TRAIN_MODE="rand_final"
# EVAL_DATA="mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev"

# for eval_data in $EVAL_DATA; do
#   CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
#     --model_class_name $MODEL_CLASS_NAME \
#     --train_data $TRAIN_DATA \
#     --train_mode $TRAIN_MODE \
#     --eval_data "${eval_data}"
# done

MODEL_CLASS_NAME="distilbert"
TRAIN_DATA="mnli"
TRAIN_MODE="rand_final"
EVAL_DATA="mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev"

for eval_data in $EVAL_DATA; do
  CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
    --model_class_name $MODEL_CLASS_NAME \
    --train_data $TRAIN_DATA \
    --train_mode $TRAIN_MODE \
    --eval_data "${eval_data}"
done
