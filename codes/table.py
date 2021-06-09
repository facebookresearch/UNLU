# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python3
# Print the table used in the paper

import pandas as pd
import argparse
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datas",
        type=str,
        default="mnli_m_dev,mnli_mm_dev,snli_dev,snli_test,anli_r1_dev,anli_r2_dev,anli_r3_dev",
        help="comma separated data names",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="roberta.large.mnli,bart.large.mnli,distilbert.mnli,infersent.mnli,convnet.mnli,bilstmproj.mnli",
        help="comma separated model names",
    )
    parser.add_argument(
        "--data_loc", type=str, default="", help="location of the output data"
    )
    parser.add_argument(
        "--rand_key",
        type=str,
        default="rand_100_p_1.0_k_0.0_stop_False_punct_True",
        help="Randomization scheme",
    )
    parser.add_argument(
        "--output_mode", type=str, default="markdown", help="markdown/latex"
    )
    args = parser.parse_args()

    models = args.models.split(",")
    datas = args.datas.split(",")

    dfs = []
    for model in models:
        for data in datas:
            p = (
                Path(args.data_loc)
                / data
                / args.rand_key
                / "outputs"
                / f"{model}.jsonl"
            )
            print(p)
            df = pd.read_json(p, lines=True)
            df["Eval Data"] = data
            dfs.append(df)

    dfs = pd.concat(dfs)

    clean_names = {
        "infersent.mnli": "InferSent",
        "convnet.mnli": "ConvNet",
        "bilstmproj.mnli": "BiLSTM",
        "roberta.large.mnli": "RoBERTa (large)",
        "bart.large.mnli": "BART (large)",
        "distilbert.mnli": "DistilBERT",
        "chinese.roberta.large.ocnli": "RoBERTa-L",
        "infersent.ocnli": "InferSent",
        "convnet.ocnli": "ConvNet",
        "bilstm.ocnli": "BiLSTM",
    }
    dfs["Model"] = dfs["Model"].apply(lambda x: clean_names[x])
    cols = [
        "Model",
        "Eval Data",
        "Original Accuracy",
        "Max Accuracy",
        "Correct > Random Percentage",
        "orig_correct_cor_mean",
        "flipped_cor_mean",
    ]
    dfs = dfs[cols]

    for cl in cols[2:]:
        dfs[cl] = np.round(dfs[cl], 3)

    if args.output_mode == "markdown":
        print(dfs.to_markdown(index=False))
    elif args.output_mode == "latex":
        print(dfs.to_latex(index=False))
    else:
        print(dfs)
