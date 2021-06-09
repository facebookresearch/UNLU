#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#

wget "https://github.com/CLUEbenchmark/OCNLI/raw/main/data/ocnli/train.50k.json"
wget "https://github.com/CLUEbenchmark/OCNLI/raw/main/data/ocnli/dev.json"
wget "https://github.com/CLUEbenchmark/OCNLI/raw/main/data/ocnli/test.json"

mv train.50k.json train.json

python ocnli_preprocess.py
cp ocnli_dev.csv ../../rnn_models/ocnli/
cp ocnli_train.csv ../../rnn_models/ocnli/
