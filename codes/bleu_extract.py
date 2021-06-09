# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

import json
import pandas as pd
import sys

from utils.common import load_jsonl

import argparse
import numpy as np
import spacy
from spacy.lang.en import English

nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.Defaults.create_tokenizer(nlp)
from tqdm.auto import tqdm
import copy
from nltk.translate.bleu_score import sentence_bleu as bleu_score


def get_bleu(sent1, sent2, n=2):
    sent1 = [w.text for w in tokenizer(sent1)]
    sent2 = [w.text for w in tokenizer(sent2)]
    weight = np.ones(n) / n
    return round(bleu_score([sent1], sent2, weight), 4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, default="")
    parser.add_argument("--data", type=str, default="mnli_m_dev")
    parser.add_argument("--model", type=str, default="roberta.large.mnli")
    parser.add_argument(
        "--rand_mode", type=str, default="rand_100_p_1.0_k_0.0_stop_False_punct_True"
    )
    args = parser.parse_args()

    mnli_m_data = load_jsonl(f"{args.loc}/{args.data}/{args.model}.jsonl")
    mnli_m_data_rand = load_jsonl(
        f"{args.loc}/{args.data}/{args.rand_mode}/{args.model}.jsonl"
    )

    pb = tqdm(total=len(mnli_m_data))
    bleu_store = {}
    for i in range(len(mnli_m_data)):
        orig_r = mnli_m_data[i]
        bleu_store[orig_r["uid"]] = []
        for row in mnli_m_data_rand[i * 100 : i * 100 + 100]:
            assert row["uid"].split("_seed")[0] == orig_r["uid"]
            brow = copy.deepcopy(row)
            for b in [2, 3, 4]:
                brow[f"premise_bleu_{b}"] = get_bleu(
                    row["premise"], orig_r["premise"], n=b
                )
                brow[f"hypothesis_bleu_{b}"] = get_bleu(
                    row["hypothesis"], orig_r["hypothesis"], n=b
                )
            bleu_store[orig_r["uid"]].append(brow)
        pb.update(1)
    pb.close()

    df = pd.DataFrame([r for k, v in bleu_store.items() for r in v])
    df["orig_uid"] = df.uid.apply(lambda x: x.split("_seed")[0])
    orig_corr = {row["uid"]: row["is_correct"] for row in mnli_m_data}
    df["orig_corr"] = df.orig_uid.apply(lambda x: orig_corr[x])
    df.to_csv(f"{args.loc}/{args.data}/{args.rand_mode}/{args.model}_bleu.csv")
    print("Done")
