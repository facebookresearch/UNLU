#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
# ## ANLI ALL
# python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/anli/r1/train.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r1_train.txt
# python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/anli/r2/train.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r2_train.txt
# python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/anli/r3/train.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r3_train.txt
# python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/anli/r1/dev.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r1_dev.txt
# python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/anli/r2/dev.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r2_dev.txt
# python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/anli/r3/dev.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r3_dev.txt
# python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/anli/r1/test.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r1_test.txt
# python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/anli/r2/test.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r2_test.txt
# python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/anli/r3/test.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r3_test.txt
# ## combined
# cat /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r1_train.txt /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r2_train.txt /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r3_train.txt >/private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_all_train.txt
# cat /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r1_dev.txt /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r2_dev.txt /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r3_dev.txt >/private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_all_dev.txt
# cat /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r1_test.txt /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r2_test.txt /private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_r3_test.txt >/private/home/koustuvs/mlp/nli_gen/anli/data/lm/anli_all_test.txt
# ## SNLI
python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/snli/train.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/snli_train.txt
python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/snli/dev.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/snli_dev.txt
python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/snli/test.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/snli_test.txt
## MNLI
python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/mnli/train.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/mnli_train.txt
python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/mnli/m_dev.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/mnli_m_dev.txt
python src/prepare_lm_data.py --inp /private/home/koustuvs/mlp/nli_gen/anli/data/build/mnli/mm_dev.jsonl --outp /private/home/koustuvs/mlp/nli_gen/anli/data/lm/mnli_mm_dev.txt
