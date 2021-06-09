#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
MODEL_CLASS_NAME="distilbert"
TRAIN_DATA="mnli"
TRAIN_MODE="orig"
EVAL_DATA="mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev"

for eval_data in $EVAL_DATA; do
	CUDA_VISIBLE_DEVICES=1 python src/nli/evaluation.py \
		--model_class_name $MODEL_CLASS_NAME \
		--train_data $TRAIN_DATA \
		--train_mode $TRAIN_MODE \
		--eval_data "${eval_data}"
done
