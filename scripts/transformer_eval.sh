#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
MODEL_TYPE=hub
MODEL_NAME=roberta.large.mnli
for eval_data in mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev; do
	python main.py --eval_data $eval_data --model_type $MODEL_TYPE --model_name $MODEL_NAME
done
wait

MODEL_TYPE=hub
MODEL_NAME=bart.large.mnli
for eval_data in mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev; do
	python main.py --eval_data $eval_data --model_type $MODEL_TYPE --model_name $MODEL_NAME
done
wait

MODEL_TYPE=hf_mnli_distilbert
MODEL_NAME=distilbert.mnli
for eval_data in mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev; do
	python main.py --eval_data $eval_data --model_type $MODEL_TYPE --model_name $MODEL_NAME
done
wait

MODEL_TYPE=rnn_infersent
MODEL_NAME=infersent.mnli
for eval_data in mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev; do
	python main.py --eval_data $eval_data --model_type $MODEL_TYPE --model_name $MODEL_NAME
done
wait

MODEL_TYPE=rnn_convnet
MODEL_NAME=convnet.mnli
for eval_data in mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev; do
	python main.py --eval_data $eval_data --model_type $MODEL_TYPE --model_name $MODEL_NAME
done
wait

MODEL_TYPE=rnn_blstmprojencoder
MODEL_NAME=bilstmproj.mnli
for eval_data in mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev; do
	python main.py --eval_data $eval_data --model_type $MODEL_TYPE --model_name $MODEL_NAME --slurm
done
wait

# Rand training
MODEL_TYPE=rnn_infersent_rand
MODEL_NAME=infersent.mnli.rand
for eval_data in mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev; do
	python main.py --eval_data $eval_data --model_type $MODEL_TYPE --model_name $MODEL_NAME
done
wait
