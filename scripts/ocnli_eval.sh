#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Configure dataset here
EVAL_DATA=ocnli_dev
RAND_TYPE=rand_100_p_1.0_k_0.0_stop_False_punct_True

## OCNLI
cd anli/

source setup.sh

echo $PYTHONPATH

# Generate original inference
python src/nli/evaluation.py --model_class_name chinese-roberta-large --train_data ocnli --train_mode orig --eval_data $EVAL_DATA
# Copy rand files
RAND_INP_LOC=data/build/rand/${EVAL_DATA}_rand.jsonl
mkdir -p data/build/rand/
cp ../data/$EVAL_DATA/$RAND_TYPE/rand.jsonl $RAND_INP_LOC
# Generate rand inference
python src/nli/evaluation.py --model_class_name chinese-roberta-large --train_data ocnli --train_mode orig --eval_data ${EVAL_DATA}_rand

# move to parent folder
cd ..
# Reformat outputs
MODEL_ORIG_LOC="anli/outputs/chinese-roberta-large/ocnli_train_orig_dev_${EVAL_DATA}/${EVAL_DATA}.jsonl"
MODEL_RAND_LOC="anli/outputs/chinese-roberta-large/ocnli_train_orig_dev_${EVAL_DATA}_rand/${EVAL_DATA}_rand.jsonl"
# New locations
MODEL_ORIG_OUTP="data/${EVAL_DATA}/chinese-roberta-large.jsonl"
MODEL_RAND_OUTP="data/${EVAL_DATA}/${RAND_TYPE}/chinese-roberta-large.jsonl"

ORIG_INP_LOC=anli/$(cat anli/data_store.json | jq -r .${EVAL_DATA}.orig)
RAND_INP_LOC=anli/${RAND_INP_LOC}

echo "Setting ORIG_INP_LOC=${ORIG_INP_LOC}"
echo "Setting RAND_INP_LOC=${RAND_INP_LOC}"

python codes/reformat.py --orig_loc $ORIG_INP_LOC --rand_loc $RAND_INP_LOC --model_orig_loc $MODEL_ORIG_LOC --model_rand_loc $MODEL_RAND_LOC --model_orig_outp $MODEL_ORIG_OUTP --model_rand_outp $MODEL_RAND_OUTP
echo "Reformatting done, computing stats..."
python main.py --eval_data $EVAL_DATA --model_type hf_ocnli_roberta --model_name chinese-roberta-large
