# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Scripts for generating randomziations

from genericpath import exists
from random import Random
from typing import List, Dict, Any
from numpy.core.records import array
import torch
import json
import pandas as pd
import copy
import random
from tqdm.auto import tqdm
from pathlib import Path

import spacy
from spacy.lang.en import English
from spacy.tokens.token import Token
import argparse
import pandas as pd
from collections import Counter
import copy
import random
from scipy.special import softmax
import argparse
import yaml
from addict import Dict

import numpy as np

import sys

# sys.path.append("nli_gen")
# sys.path.append("nli_gen/anli/")
# sys.path.append("nli_gen/anli/src")

# from anli.src.nli.training import config, MODEL_CLASSES, id2label, registered_path
# from anli.src.nli.evaluation import (
#     model_base_loc,
#     model_store,
#     data_base_loc,
#     data_store,
# )
from codes.common import load_json, load_jsonl, save_jsonl


random.seed(42)


class CustomToken:
    def __init__(self, token, whitespace=""):
        self.token = token
        self.whitespace = whitespace

    def get_text(self):
        token = copy.deepcopy(self.token)
        token = token + self.whitespace
        return token

    def __str__(self):
        return self.token


def has_pre_space(token):
    if token.i == 0:
        return False
    if token.nbor(-1).whitespace_:
        return True
    else:
        return False


def has_space(token):
    if type(token) == spacy.tokens.token.Token:
        return token.whitespace_
    elif type(token) == CustomToken:
        return token.whitespace


def revtok(tokens):
    retok = "".join([str(tokens[i]) + has_space(tokens[i]) for i in range(len(tokens))])
    # assert retok == str(tokens)
    return retok


def derandomize(sent, pos, tokenizer, separator=""):
    toks = [str(s) for s in tokenizer(sent)]
    return separator.join(toks[i] for i in np.argsort(pos))


def fixed_shuffle_numpy(lst, mask, percent=1.0):
    to_permute = [i for i, m in enumerate(mask) if m]
    num_permute = int(len(to_permute) * percent)
    num_to_fix = len(to_permute) - num_permute
    if num_to_fix > 0:
        to_fix_ids = random.sample(to_permute, num_to_fix)
        mask = [True if mask[i] or i in to_fix_ids else False for i in range(len(mask))]
    # mask = [True if mask[i] or random.uniform(0,1) > percent else False for i,e in enumerate(mask)]
    unfrozen_indices = [i for i, e in enumerate(lst) if not mask[i]]
    unfrozen_set = lst[unfrozen_indices]
    unfrozen_set_og = copy.deepcopy(unfrozen_set)
    if len(unfrozen_set) > 1:
        while True:
            random.shuffle(unfrozen_set)
            same_pos = sum(
                [
                    1
                    for i, e in enumerate(unfrozen_set)
                    if unfrozen_set[i] == unfrozen_set_og[i]
                ]
            )
            if same_pos == 0:
                break
    lst[unfrozen_indices] = unfrozen_set


def randomize(
    sent,
    retain_stop=False,
    retain_punct=False,
    keep_order=0.0,
    percent=1.0,
    seed=42,
    lang="en",
    tokenizer=None,
):
    """
    Main randomization function

    Args:
        sent: sentence to randomize (str)
        retain_stop: if true, retain stop words in their own positions
        retain_punct: if true, retain punctuations in their own positions
        keep_order (float) percentage of words to keep unshuffled together
        percent (float): percentage of tokens to randomize (Default: 1.0)
        lang: "en" for English, "zh" for Chinese
        tokenizer: Spacy tokenizer
        seed: seed for reproducibility
    """
    random.seed(seed)
    tok = tokenizer(sent)
    kept_order = False
    num_to_keep = int(len(tok) * keep_order)
    if keep_order > 0 and num_to_keep > 0:
        if num_to_keep == len(tok):
            raise AssertionError("Cannot shuffle if all words are kept")
        start_id_to_keep = random.choice(range(0, len(tok) - num_to_keep))
        end_id_to_keep = start_id_to_keep + num_to_keep
        new_tok = []
        build = []
        for ti in range(len(tok)):
            if ti >= start_id_to_keep and ti < end_id_to_keep:
                build.append(tok[ti])
            else:
                new_tok.append(tok[ti])
            if ti == end_id_to_keep:
                new_tok.append(
                    CustomToken(revtok(build), whitespace=build[-1].whitespace_)
                )
        tok = new_tok
        kept_order = True
    masks = []
    masks.append(np.zeros(len(tok)).astype(bool))
    if retain_punct:
        masks.append(
            np.array(
                [
                    t.is_punct if type(t) == spacy.tokens.token.Token else False
                    for t in tok
                ]
            )
        )
    if retain_stop:
        masks.append(
            np.array(
                [
                    t.is_stop if type(t) == spacy.tokens.token.Token else False
                    for t in tok
                ]
            )
        )
    mask = np.any(np.stack(masks), axis=0)
    indx = np.arange(len(tok))
    fixed_shuffle_numpy(indx, mask, percent)
    new_tok = [tok[i] for i in indx]
    if lang == "en":
        separator = " "
    else:
        separator = ""
    new_sent = separator.join([str(t) for t in new_tok])
    if lang == "en":
        # removing extra space
        new_sent = separator.join(new_sent.split())
    # not possible for chinese as the tokenization words matter on randomization
    # de_rand = derandomize(new_sent, indx, tokenizer)
    # assert de_rand == sent
    # percent changed
    ch = sum([1 for i, p in enumerate(indx) if i != p and not mask[i]]) / sum(
        [1 for m in mask if not m]
    )
    return new_sent, indx, ch, mask, kept_order


def single_predict(model, tokenizer, premise, hypothesis, max_length=156):
    with torch.no_grad():
        inp = tokenizer.encode_plus(
            premise,
            hypothesis,
            max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
        )
        outputs = model(
            torch.tensor(inp["input_ids"]).unsqueeze(0),
            attention_mask=torch.tensor(inp["attention_mask"]).unsqueeze(0),
        )
        return id2label[torch.max(outputs[0], 1)[1].item()], outputs[0]


def compute_stats(result):
    cor_pred, o_pred = result
    count = len(o_pred)
    correct = sum([1 for row in o_pred if row["label"] == row["predicted_label"]])
    correct_acc = correct / count
    rand_acc = len(cor_pred) / count
    print(f"Original accuracy : {correct_acc}")
    print(f"Random input accuracy : {rand_acc}")
    return {"orig_acc": correct_acc, "rand_acc": rand_acc}


def prepare_data(
    data,
    outp: Path,
    num_tries=100,
    sent1_label="premise",
    sent2_label="hypothesis",
    target_label="label",
    index_label="uid",
    percent=1.0,
    keep_order=0.0,
    retain_stop=False,
    retain_punct=True,
    save_data=True,
    rebuild=True,
    keep_premise=False,
    lang="en",
):
    """Prepare dataset for randomization tests

    Args:
        data ([type]): [description]
        outp ([type]): [description]
        num_tries (int, optional): [description]. Defaults to 100.
        sent1_label (str, optional): [description]. Defaults to "premise".
        sent2_label (str, optional): [description]. Defaults to "hypothesis".
        target_label (str, optional): [description]. Defaults to "label".
        index_label (str, optional): [description]. Defaults to "uid".
        percent (float, optional): [description]. Defaults to 1.0.
        keep_order (float, optional): [description]. Defaults to 0.0.
        retain_stop (bool, optional): [description]. Defaults to False.
        retain_punct (bool, optional): [description]. Defaults to True.
    """
    if not rebuild:
        if outp.exists():
            print(
                "Randomized data exists and rebuild set to False, loading the same data..."
            )
            rand_eval = load_jsonl(outp)
            return rand_eval
        else:
            print(
                f"Rebuild set to False but file not present at {outp}, building from scratch..."
            )
    ## Build tokenizer
    if lang == "en":
        nlp = English()
    elif lang == "cn":
        nlp = spacy.load("zh_core_web_lg")
    else:
        raise NotImplementedError(f"Tokenizer for Language={lang} not implement")

    tokenizer = nlp.Defaults.create_tokenizer(nlp)
    pb = tqdm(total=len(data))
    rand_eval = []
    for row in data:
        ## TODO: Compute the number of possible permutations
        for seed in range(num_tries):
            if keep_premise:
                # keep the same sentence
                s1 = row[sent1_label]
                s1_idx = np.arange(len(tokenizer(s1)))
                s1_kept_order = False
            else:
                s1, s1_idx, _, _, s1_kept_order = randomize(
                    row[sent1_label],
                    retain_stop=retain_stop,
                    retain_punct=retain_punct,
                    percent=percent,
                    keep_order=keep_order,
                    seed=seed,
                    lang=lang,
                    tokenizer=tokenizer,
                )
            s2, s2_idx, _, _, s2_kept_order = randomize(
                row[sent2_label],
                retain_stop=retain_stop,
                retain_punct=retain_punct,
                percent=percent,
                keep_order=keep_order,
                seed=seed,
                lang=lang,
                tokenizer=tokenizer,
            )
            rand_eval.append(
                {
                    index_label: str(row[index_label]) + f"_seed_{seed}",
                    sent1_label: s1,
                    sent2_label: s2,
                    f"{sent1_label}_pos": s1_idx.tolist(),
                    f"{sent2_label}_pos": s2_idx.tolist(),
                    "seed": seed,
                    target_label: row[target_label],
                    f"{sent1_label}_kept_order": s1_kept_order,
                    f"{sent2_label}_kept_order": s2_kept_order,
                }
            )
        pb.update(1)
    pb.close()
    if save_data:
        print("Data generation done. Saving ...")
        save_jsonl(rand_eval, outp)
    return rand_eval


## Classes


class SentenceDataRandomizer:
    def __init__(self, args, data_name="") -> None:
        self.args = args
        self.data_name = data_name
        self.orig_data_loc = ""
        self.orig_data = []
        self.rand_data_loc = ""
        self.rand_data_file = ""
        self.rand_data = []
        self.orig_data_outp_file = ""
        self.orig_data_outp = []
        self.rand_data_outp_file = ""
        self.rand_data_outp = []
        self.sent1_label = ""
        self.sent2_label = ""
        self.index_label = ""
        self.target_label = ""
        self.orig_data_copy = (
            Path(self.args.outp_path) / self.args.eval_data / "orig.jsonl"
        )
        self.data_loc = Path(args.outp_path) / args.eval_data
        folder_name = (
            f"rand_{args.data_prep_config.num_tries}"
            + f"_p_{args.data_prep_config.percent}"
            + f"_k_{args.data_prep_config.keep_order}"
            + f"_stop_{args.data_prep_config.retain_stop}"
            + f"_punct_{args.data_prep_config.retain_punct}"
        )
        if args.data_prep_config.keep_premise:
            folder_name += f"_keep_premise"
        if len(args.rand_data_folder) > 0:
            folder_name = args.rand_data_folder
        self.rand_data_loc = self.data_loc / folder_name
        self.rand_data_loc.mkdir(parents=True, exist_ok=True)
        self.rand_data_file = self.rand_data_loc / "rand.jsonl"

    def load_data(self) -> None:
        """Load a jsonl dataset. Override if the dataset is not jsonl"""
        self.orig_data = load_jsonl(self.orig_data_loc)
        if not self.orig_data_copy.exists():
            save_jsonl(self.orig_data, self.orig_data_copy)

    def randomize_data(self) -> None:
        print(f"Preparing data for {self.data_name}")
        self.rand_data = prepare_data(
            self.orig_data,
            self.rand_data_file,
            sent1_label=self.sent1_label,
            sent2_label=self.sent2_label,
            target_label=self.target_label,
            **self.args.data_prep_config,
        )

    def load_predictions(self, model_name, pred_type=""):
        if pred_type == "rand":
            self.rand_data_outp_file = self.rand_data_loc / f"{model_name}.jsonl"
            print(f"Looking for file {self.rand_data_outp_file}")
            if self.rand_data_outp_file.exists():
                print("File exists, loading predictions...")
                self.rand_data_outp = load_jsonl(self.rand_data_outp_file)
                return True
            else:
                print("File does not exist, building predictions ..")
                return False
        else:
            self.orig_data_outp_file = self.data_loc / f"{model_name}.jsonl"
            print(f"Looking for file {self.orig_data_outp_file}")
            if self.orig_data_outp_file.exists():
                print("File exists, loading predictions...")
                self.orig_data_outp = load_jsonl(self.orig_data_outp_file)
                return True
            else:
                print("File does not exist, building predictions ..")
                return False

    def save_predictions(self, preds, model_name, pred_type=""):
        """Save predictions

        Args:
            preds ([type]): [description]
            model_name ([type]): [description]
            pred_type (str, optional): [description]. Defaults to "".
        """
        if pred_type == "rand":
            self.rand_data_outp_file = self.rand_data_loc / f"{model_name}.jsonl"
            self.rand_data_outp = preds
            save_jsonl(preds, self.rand_data_outp_file)
        else:
            self.orig_data_outp_file = self.data_loc / f"{model_name}.jsonl"
            self.orig_data_outp = preds
            save_jsonl(preds, self.orig_data_outp_file)

    def print_stats(self, model_name, agg_out):
        """Compare stats with the aggregated output

        Args:
            model_name ([type]): [description]
            agg_out ([type]): [description]
        """
        stats = {}

        def cprint(key, val):
            stats[key] = val
            print(f"{key} : {val}")

        orig_out = pd.DataFrame(self.orig_data_outp)
        if model_name == "chinese-roberta-large":
            orig_out.uid = orig_out.uid.astype(int)
        # orig_acc = len(orig_out[orig_out.is_correct]) / len(orig_out)
        cprint("Model", model_name)
        print("========================")
        # cprint("Original Accuracy", orig_acc)
        rand_out = pd.DataFrame(agg_out)

        if self.args.hyp_distinct_perm:
            print(
                f"Warning: reducing eval data from {len(rand_out)} to {len(rand_out[rand_out.hyp_distinct_perm])}"
            )
            rand_out = rand_out[rand_out.hyp_distinct_perm]
        if self.args.prem_distinct_perm:
            print(
                f"Warning: reducing eval data from {len(rand_out)} to {len(rand_out[rand_out.prem_distinct_perm])}"
            )
            rand_out = rand_out[rand_out.prem_distinct_perm]
        rand_acc = len(rand_out[rand_out.exists]) / len(rand_out)
        orig_out_subset = orig_out[orig_out.uid.isin(rand_out.uid)]
        orig_acc = len(orig_out_subset[orig_out_subset.is_correct]) / len(
            orig_out_subset
        )
        cprint("Original Accuracy", orig_acc)
        print(f"Chosen number: {len(rand_out)}")
        cprint("Max Accuracy", rand_acc)
        print("==============")
        print("Originally Correct")
        print("-------------")
        for mode in ["orig_correct", "flipped"]:
            corr_num = len(rand_out[(rand_out.exists) & (rand_out.pred_mode == mode)])
            corr_per = np.round(
                len(rand_out[(rand_out.exists) & (rand_out.pred_mode == mode)])
                / len(rand_out),
                3,
            )
            cprint(f"{mode}_num", corr_num)
            cprint(f"{mode}_percent", corr_per)
            print(f"mode: {mode}")
            for target in ["e", "n", "c"]:
                t_mean = rand_out[
                    (rand_out.pred_mode == mode)
                    & (rand_out.label == target)
                    & rand_out.exists
                ][f"prob_{target}_mean"].mean()
                t_std = rand_out[
                    (rand_out.pred_mode == mode)
                    & (rand_out.label == target)
                    & rand_out.exists
                ][f"prob_{target}_mean"].std()
                cprint(
                    f"{mode}_{target}",
                    f"{np.round(t_mean, 3)} +/- {np.round(t_std, 3)}",
                )
            cor_mean = rand_out[
                (rand_out.pred_mode == mode) & (rand_out.exists)
            ].cor_mean.mean()
            cprint(f"{mode}_cor_mean", cor_mean)
            print("==============")
        cprint(
            "Majority Acc",
            len(rand_out[rand_out.is_correct_rand_majority]) / len(rand_out),
        )
        cprint(
            "Correct > Random Percentage",
            len(rand_out[rand_out.is_correct_gt_random]) / len(rand_out),
        )
        rand_out.uid = rand_out.uid.astype(str)
        orig_out.uid = orig_out.uid.astype(str)
        jdf = pd.merge(rand_out, orig_out, on="uid")
        cprint(
            "Combined Acc",
            len(jdf[(jdf.is_correct) | (jdf.is_correct_rand_majority)]) / len(jdf),
        )
        print("-------------")
        ## Save stats
        outp_path = Path(self.rand_data_loc) / "outputs"
        outp_path.mkdir(exist_ok=True, parents=True)
        save_jsonl([stats], outp_path / f"{model_name}.jsonl")
        jdf.to_csv(outp_path / f"{model_name}_filtered_rand_outs.csv")


class ANLI(SentenceDataRandomizer):
    def __init__(self, args) -> None:
        super().__init__(args, args.eval_data)
        wd = Path(".").resolve()
        self.orig_data_loc = Path(wd) / args.data_path / args[args.eval_data].orig_path
        self.sent1_label = args.mnli_m_dev.sent1_label
        self.sent2_label = args.mnli_m_dev.sent2_label
        self.index_label = args.mnli_m_dev.index_label
        self.target_label = args.mnli_m_dev.target_label


class RTE(SentenceDataRandomizer):
    def __init__(self, args) -> None:
        super().__init__(args, args.eval_data)
        wd = Path(".").resolve()
        self.orig_data_loc = Path(wd) / args[args.eval_data].orig_path
        self.sent1_label = args.rte_dev.sent1_label
        self.sent2_label = args.rte_dev.sent2_label
        self.index_label = args.rte_dev.index_label
        self.target_label = args.rte_dev.target_label
        self.orig_data_copy = (
            Path(self.args.outp_path) / self.args.eval_data / "orig.jsonl"
        )

    def load_data(self) -> None:
        """Load a jsonl dataset."""
        sent1 = []
        sent2 = []
        label = []
        with open(self.orig_data_loc / "dev.raw.input0", "r") as fp:
            for line in fp:
                sent1.append(line.rstrip())
        with open(self.orig_data_loc / "dev.raw.input1", "r") as fp:
            for line in fp:
                sent2.append(line.rstrip())
        with open(self.orig_data_loc / "dev.label", "r") as fp:
            for line in fp:
                label.append(line.rstrip())
        self.orig_data = [
            {
                "uid": i,
                self.sent1_label: sent1[i],
                self.sent2_label: sent2[i],
                self.target_label: label[i],
            }
            for i in range(len(sent1))
        ]

        if not self.orig_data_copy.exists():
            save_jsonl(self.orig_data, self.orig_data_copy)


def main_prep(args):
    mnli_m_dev = ANLI(args)
    mnli_m_dev.load_data()
    mnli_m_dev.randomize_data()
    print("Done")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", default="mnli_m_dev", type=str, help="eval data")
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="location of config file"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config = yaml.load(open(args.config))
    config["eval_data"] = args.eval_data
    main_prep(Dict(config))
