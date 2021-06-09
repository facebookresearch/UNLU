#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#

wget "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
unzip multinli_1.0.zip
python mnli_preprocess.py
cp processed_mnli_dev_glove.csv ../../rnn_models/mnli/
cp processed_mnli_train_glove.csv ../../rnn_models/mnli/
