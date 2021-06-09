# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Eval using FAIRSEQ trained models
from codes.rnn_models import InferSent
import torch
from tqdm.auto import tqdm
import copy
from collections import Counter
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
import pandas as pd
import random
from codes.models import FairSeqModel, NLI_Model, HubModel, HFModel

# from fairseq.models.roberta import RobertaModel

random.seed(42)
from codes.word_randomization import RTE, SentenceDataRandomizer, ANLI


def predict_dataset(
    model: NLI_Model,
    data,
    batch_size=4,
    sent1_label="premise",
    sent2_label="hypothesis",
    target_label="label",
    index_label="uid",
    **kwargs,
):
    # ncorrect, nsamples = 0, 0
    batches = model.prepare_batches(
        data,
        batch_size=batch_size,
        sent1_label=sent1_label,
        sent2_label=sent2_label,
        target_label=target_label,
    )
    print("predicting ..")
    pb = tqdm(total=len(batches))
    preds = []
    pred_logits = []
    outp = []
    with torch.no_grad():
        for batch, meta in batches:
            prediction, prediction_logits = model.predict_batch(batch)
            preds.append(prediction)
            pred_logits.append(prediction_logits)
            pb.update(1)
    pb.close()
    print("Collecting ...")
    pb = tqdm(total=len(batches))
    for bi, (_, meta) in enumerate(batches):
        for i, row in enumerate(meta):
            if type(model.label_fn) == dict:
                prediction_label = model.label_fn[preds[bi][i]]
            else:
                prediction_label = model.label_fn(preds[bi][i])
            outp.append(
                {
                    index_label: row[0],
                    sent1_label: row[1],
                    sent2_label: row[2],
                    "orig_label": row[3],
                    "predicted_label": prediction_label,
                    "is_correct": row[3] == prediction_label,
                    "logits": pred_logits[bi][i],
                }
            )
            pb.update(1)
    pb.close()
    return outp


def is_unique_permutation(rows, pos_label="hypothesis_pos"):
    return len(set(["-".join([str(x) for x in r[pos_label]]) for r in rows])) == len(
        rows
    )


def compute_rand_acc(
    data,
    data_out,
    rand_inp,
    rand_out,
    entailment_id=0,
    contradiction_id=1,
    neutral_id=2,
):
    """Compute the prediction stats

    Args:
        data ([type]): original data
        data_out ([type]): original data model output
        rand_inp ([type]): randomized data
        rand_out ([type]): randomized data model output
        entailment_id (int, optional): [description]. Defaults to 0.
        contradiction_id (int, optional): [description]. Defaults to 1.
        neutral_id (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    num_corr = 0
    num_corr_rand = 0  # if cor_mean > 0.33, then increment this value; This means the correct label is chosen more than random. Does not indicate whether correct label is chosen max number of times.

    pb = tqdm(total=len(data))
    result = []
    total_count = 0
    total_hits = 0
    for i, row in enumerate(data):
        is_r = False
        crow = copy.deepcopy(row)
        start = i * 100
        rand_rows = rand_out[start : start + 100]
        rand_inp_rows = rand_inp[start : start + 100]
        gold_label = crow["label"]
        rand_labels = [c["predicted_label"] for c in rand_rows]
        # assert crow['uid'] == data_out[i]['uid']
        crow["orig_pred"] = data_out[i]["predicted_label"]
        crow["sample_premise"] = ""
        crow["sample_hypothesis"] = ""
        crow["prob_e_mean"] = 0
        crow["prob_e_std"] = 0
        crow["prob_n_mean"] = 0
        crow["prob_n_std"] = 0
        crow["prob_c_mean"] = 0
        crow["prob_c_std"] = 0
        crow["hyp_distinct_perm"] = is_unique_permutation(
            rand_inp_rows, "hypothesis_pos"
        )
        crow["prem_distinct_perm"] = is_unique_permutation(rand_inp_rows, "premise_pos")
        ct = Counter(rand_labels)
        ct_label = {"e": 0, "n": 0, "c": 0}
        for k, v in ct.items():
            ct_label[k] = v
        total_hits += ct[gold_label]
        total_count += len(rand_labels)
        if gold_label in rand_labels:
            crow["exists"] = True
            num_corr += 1
            crow["cor_mean"] = ct[gold_label] / len(rand_labels)
            ids = [p for p in range(100) if rand_labels[p] == gold_label]
            # calculate avg probabilities here
            sf = np.array([softmax(rand_rows[id]["logits"]) for id in ids])
            crow["prob_e_mean"] = np.mean(sf[:, entailment_id])
            crow["prob_e_std"] = np.std(sf[:, entailment_id])
            crow["prob_n_mean"] = np.mean(sf[:, neutral_id])
            crow["prob_n_std"] = np.std(sf[:, neutral_id])
            crow["prob_c_mean"] = np.mean(sf[:, contradiction_id])
            crow["prob_c_std"] = np.std(sf[:, contradiction_id])
            crow["prob_entropy"] = entropy(sf, axis=1).mean()
            ids = random.choice(ids) + i * 100
            # sanity check correct record
            if "_seed" in rand_inp[ids]["uid"]:
                assert rand_inp[ids]["uid"].split("_seed")[0] == str(crow["uid"])
            else:
                assert rand_inp[ids]["uid"] == crow["uid"]
            crow["sample_premise"] = rand_inp[ids]["premise"]
            crow["sample_hypothesis"] = rand_inp[ids]["hypothesis"]
            crow["seed_id"] = ids
        else:
            crow["exists"] = False
            crow["cor_mean"] = 0
        if crow["cor_mean"] > (1 / 3):
            num_corr_rand += 1
            crow["is_correct_gt_random"] = True
        else:
            crow["is_correct_gt_random"] = False
        crow["rand_class_count"] = ",".join([f"{k}:{v}" for k, v in ct_label.items()])
        crow["rand_majority"] = max(ct_label, key=ct_label.get)
        crow["is_correct_rand_majority"] = crow["rand_majority"] == crow["label"]
        result.append(crow)
        pb.update(1)
    pb.close()
    result = pd.DataFrame(result)
    print(f"Max accuracy : {num_corr / len(data)}")
    print(f"Full accuracy : {total_hits / total_count}")
    print(f"Correct>Rand Percent: {num_corr_rand / len(data)}")
    result.loc[(result.orig_pred == result.label), "pred_mode"] = "orig_correct"
    result.loc[(result.orig_pred != result.label), "pred_mode"] = "flipped"
    return result


DATA_M_CLASS = {
    "mnli_m_dev": ANLI,
    "mnli_mm_dev": ANLI,
    "snli_dev": ANLI,
    "snli_test": ANLI,
    "anli_r1_dev": ANLI,
    "anli_r2_dev": ANLI,
    "anli_r3_dev": ANLI,
    "rte_dev": RTE,
    "qqp_dev": RTE,
    "qnli_dev": RTE,
    "ocnli_dev": ANLI,
}


def main_eval(args):
    print("Loading model ...")
    if args.model_type == "hub":
        model = HubModel(args[args.model_type])
    elif "fairseq" in args.model_type:
        model = FairSeqModel(args[args.model_type])
    elif args.model_type.startswith("hf_"):
        model = HFModel(args[args.model_type])
    elif args.model_type.startswith("rnn_"):
        model = InferSent(args[args.model_type])
    else:
        raise AssertionError("check model type")

    data_class = DATA_M_CLASS[args.eval_data]
    ## Load data
    data = data_class(args)
    data.load_data()
    data.randomize_data()
    model_name = args[args.model_type].model_name
    ## Predict original dataset
    if not data.load_predictions(model_name=model_name, pred_type="orig"):
        outp = predict_dataset(
            model, data.orig_data, args.batch_size, **args[args.eval_data]
        )
        data.save_predictions(outp, model_name, pred_type="orig")
    ## Predict random dataset
    if not data.load_predictions(model_name=model_name, pred_type="rand"):
        outp = predict_dataset(
            model, data.rand_data, args.batch_size, **args[args.eval_data]
        )
        data.save_predictions(outp, model_name, pred_type="rand")
    ## Compute Stats
    id_map = (
        args[args.model_type].id_map
        if "id_map" in args[args.model_type]
        else args[args.eval_data].id_map
    )
    result = compute_rand_acc(
        data.orig_data,
        data.orig_data_outp,
        data.rand_data,
        data.rand_data_outp,
        **id_map,
    )

    data.print_stats(model_name, result)
    result.to_csv(data.rand_data_loc / "outputs" / f"{model_name}.csv")

    print("Done")
