#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
MODEL_NAME=bilstmproj.mnli
for randz in 0.0; do
	for eval_data in mnli_m_dev mnli_mm_dev snli_dev snli_test anli_r1_dev anli_r2_dev anli_r3_dev; do
		python codes/bleu_extract.py --data $eval_data --model $MODEL_NAME --rand_mode "rand_100_p_1.0_k_${randz}_stop_False_punct_True" &
	done
	wait
done
wait
