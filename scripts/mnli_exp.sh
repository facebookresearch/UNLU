#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Generate MNLI_M rand data
python code/word_randomization.py --eval_data mnli_m_dev --outp build/mnli/rand_cont_0_seeds_m_dev.jsonl
## Evaluate on rand data
