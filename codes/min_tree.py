# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Compute the Min Tree hypothesis
import argparse
import spacy
import json
from json import JSONEncoder
import pandas as pd
from tqdm.auto import tqdm
import copy
from collections import Counter
import numpy as np
import pickle

import ray

ray.init(num_cpus=24)

import random
from pathlib import Path

random.seed(42)

registered_jsonabl_classes = {}

# Some Jsonable classes, for easy json serialization.


def register_class(cls):
    global registered_jsonabl_classes
    if cls not in registered_jsonabl_classes:
        registered_jsonabl_classes.update({cls.__name__: cls})


class JsonableObj(object):
    pass


class JsonableObjectEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, JsonableObj):
            d = {"_jcls_": type(o).__name__}
            d.update(vars(o))
            return d
        else:
            return super().default(o)


def unserialize_JsonableObject(d):
    global registered_jsonabl_classes
    classname = d.pop("_jcls_", None)
    if classname:
        cls = registered_jsonabl_classes[classname]
        obj = cls.__new__(cls)  # Make instance without calling __init__
        for key, value in d.items():
            setattr(obj, key, value)
        return obj
    else:
        return d


def json_dumps(item):
    return json.dumps(item, cls=JsonableObjectEncoder)


def json_loads(item_str):
    return json.loads(item_str, object_hook=unserialize_JsonableObject)


def save_jsonl(d_list, filename):
    with open(filename, encoding="utf-8", mode="w") as out_f:
        for item in d_list:
            out_f.write(
                json.dumps(item, cls=JsonableObjectEncoder, ensure_ascii=False) + "\n"
            )


def load_jsonl(filename, debug_num=None):
    d_list = []
    with open(filename, encoding="utf-8", mode="r") as in_f:
        for line in tqdm(in_f):
            item = json.loads(line.strip(), object_hook=unserialize_JsonableObject)
            d_list.append(item)
            if debug_num is not None and 0 < debug_num == len(d_list):
                break

    return d_list


univ_pos = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "AUX": 3,
    "CCONJ": 4,
    "DET": 5,
    "INTJ": 6,
    "NOUN": 7,
    "NUM": 8,
    "PART": 9,
    "PRON": 10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM": 14,
    "VERB": 15,
    "X": 16,
}

univ_pos_list = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]

## utility functions
def comp_mini_tree(sent, pos, radius=5):
    sent_tags = [w.pos_ for w in sent]
    del sent_tags[pos]
    start = pos - radius
    start = 0 if start < 0 else start
    tokens = sent_tags[start : pos + radius]
    return Counter(tokens)


def comp_tree_sentence(nlp, sent, word_tree, radius=5):
    sent = nlp(sent)
    for wi, word in enumerate(sent):
        tree = comp_mini_tree(sent, wi, radius=radius)
        w = word.lower_
        if w not in word_tree:
            word_tree[w] = tree
        else:
            word_tree[w] += tree
    return word_tree


def get_vocab_probs(nlp, prem, hyp=None, radius=1):
    t_w = {}
    t_w = comp_tree_sentence(nlp, prem, t_w, radius=radius)
    if hyp:
        t_w = comp_tree_sentence(nlp, hyp, t_w, radius=radius)
    t_vocab_probs = {}
    for word, ct in t_w.items():
        prob = np.zeros(len(univ_pos_list))
        tot = sum([v for k, v in ct.items() if k in univ_pos_list])
        for k, v in ct.items():
            if k in univ_pos_list:
                prob[univ_pos_list.index(k)] = v / tot
        t_vocab_probs[word] = prob
    return t_vocab_probs


def get_word_overlap(word, vp, train_vocab, train_vocab_probs, k=3):
    if word not in train_vocab:
        return 0
    ids = np.argpartition(vp[word], -k)[-k:]
    train_ids = np.argpartition(train_vocab_probs[train_vocab.index(word)], -k)[-k:]
    c_id = len(set(ids).intersection(train_ids))
    return c_id / len(ids)


def get_avg_overlap(nlp, sent, train_vocab, train_vocab_probs, k=3):
    probs = get_vocab_probs(nlp, sent, radius=k // 2)
    overlap = []
    for word, _ in probs.items():
        overlap.append(
            get_word_overlap(word, probs, train_vocab, train_vocab_probs, k=k)
        )
    return np.mean(overlap)


def get_min_tree_overlap(id, inp_file, outp_pre, vocab_file, k):
    nlp = spacy.load("en_core_web_md")
    rows = []
    r = {}
    all_rows = load_jsonl(inp_file)
    orig_row = all_rows[0]
    rand_rows = all_rows[1:]
    vocab_store = pickle.load(open(vocab_file, "rb"))
    train_vocab = vocab_store["vocab"]
    train_vocab_probs = vocab_store["probs"]

    r[f"premise_ov_{k}"] = get_avg_overlap(
        nlp, orig_row["premise"], train_vocab, train_vocab_probs, k=k
    )
    r[f"hypothesis_ov_{k}"] = get_avg_overlap(
        nlp, orig_row["hypothesis"], train_vocab, train_vocab_probs, k=k
    )
    r["orig_correct"] = orig_row["is_correct"]
    r["num_correct"] = sum([1 for p in rand_rows if p["is_correct"]]) / 100
    for ji, j in enumerate(rand_rows):
        rp = copy.deepcopy(r)
        rp[f"rand_premise_ov_{k}"] = get_avg_overlap(
            nlp, j["premise"], train_vocab, train_vocab_probs, k=k
        )
        rp[f"rand_hypothesis_ov_{k}"] = get_avg_overlap(
            nlp, j["hypothesis"], train_vocab, train_vocab_probs, k=k
        )
        rp["rand_is_correct"] = j["is_correct"]
        rp["rand_id"] = ji
        rp["uid"] = orig_row["uid"]
        rows.append(rp)
    # Save
    save_jsonl(rows, str(outp_pre) + f"_{id}.jsonl")


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute_train_stats",
        action="store_true",
        help="Compute the training stats to compare",
    )
    parser.add_argument(
        "--saved_model", default="data/mnli_m_dev/min_tree_train.pkl", type=str,
    )
    parser.add_argument("-k", "--overlap_k", default=2, type=int)
    parser.add_argument("--loc", type=str, default="data/")
    parser.add_argument("--data", type=str, default="mnli_m_dev")
    parser.add_argument("--model", type=str, default="roberta.large.mnli")
    parser.add_argument(
        "--rand_mode", type=str, default="rand_100_p_1.0_k_0.0_stop_False_punct_True"
    )
    parser.add_argument("--temp_loc", type=str, default="data/tmp")
    parser.add_argument("--mp", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    mnli_m_train = load_jsonl(f"anli/data/build/mnli/train.jsonl")
    mnli_m_data = load_jsonl(f"{args.loc}/{args.data}/{args.model}.jsonl")
    mnli_m_data_rand = load_jsonl(
        f"{args.loc}/{args.data}/{args.rand_mode}/{args.model}.jsonl"
    )

    ids = list(range(len(mnli_m_data)))
    if args.debug:
        ids = ids[:24]

    inp_paths = []
    k = args.overlap_k

    # train the statistics
    if args.compute_train_stats:
        print("Computing word trees")
        nlp = spacy.load("en_core_web_md")
        wt = {}
        radius = k // 2
        pb = tqdm(total=len(mnli_m_train))
        for row in mnli_m_train:
            wt = comp_tree_sentence(nlp, row["premise"], wt, radius=radius)
            wt = comp_tree_sentence(nlp, row["hypothesis"], wt, radius=radius)
            pb.update(1)
        pb.close()

        print("Computing vocab ..")
        vocab = []
        vocab_probs = []
        pb = tqdm(total=len(wt))
        for word, ct in wt.items():
            vocab.append(word)
            prob = np.zeros(len(univ_pos_list))
            tot = sum([v for k, v in ct.items() if k in univ_pos_list])
            for k, v in ct.items():
                if k in univ_pos_list:
                    prob[univ_pos_list.index(k)] = v / tot
            vocab_probs.append(prob)
            pb.update(1)
        pb.close()

        vocab_probs = np.stack(vocab_probs)
        print("Saving vocab ...")
        pickle.dump(
            {"vocab": vocab, "probs": vocab_probs}, open(args.saved_model, "wb"),
        )

    # Save the temp files
    Path(args.temp_loc).mkdir(parents=True, exist_ok=True)
    for i in ids:
        orig_row = mnli_m_data[i]
        rand_rows = mnli_m_data_rand[i * 100 : i * 100 + 100]
        all_data = [orig_row] + rand_rows
        inp_path = Path(args.temp_loc) / f"{args.data}_{args.model}_inp_p_{i}.jsonl"
        if not inp_path.exists():
            save_jsonl(all_data, inp_path)
        inp_paths.append(inp_path)

    outp_pre = Path(args.temp_loc) / f"{args.data}_{args.model}_outp_p"

    if args.mp:
        get_min_tree_overlap_r = ray.remote(get_min_tree_overlap)
        obj_ids = [
            get_min_tree_overlap_r.remote(
                i, inp_paths[i], outp_pre, args.saved_model, k
            )
            for i in ids
        ]
        for x in tqdm(to_iterator(obj_ids), total=len(obj_ids)):
            pass
    else:
        pb = tqdm(total=len(ids))
        for i in ids:
            get_min_tree_overlap(i, inp_paths[i], outp_pre, args.saved_model, k)
            pb.update(1)
        pb.close()

    print("Done!")
    print("Collecting ...")

    ## Collect
    pb = tqdm(total=len(ids))
    rows = []
    for i in ids:
        rows.extend(load_jsonl(str(outp_pre) + f"_{i}.jsonl"))
        pb.update(1)
    pb.close()

    df = pd.DataFrame(rows)
    df[f"ratio_{k}"] = (
        (df[f"rand_premise_ov_{k}"] + df[f"rand_hypothesis_ov_{k}"]) / 2
    ) / ((df[f"premise_ov_{k}"] + df[f"hypothesis_ov_{k}"]) / 2)
    df.to_csv(
        Path(args.loc) / args.data / args.rand_mode / f"{args.model}_min_tree_{k}.csv"
    )
    print("Done")
