# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path

import config
from flint.data_utils.fields import RawFlintField, LabelFlintField, ArrayIndexFlintField
from utils import common, list_dict_data_tool, save_tool
from src.nli.training import (
    MODEL_CLASSES,
    registered_path,
    build_eval_dataset_loader_and_sampler,
    NLITransform,
    FlippedNLITransform,
    NLIDataset,
    count_acc,
    evaluation_dataset,
    eval_model,
    nli_label2index,
)

import torch
from transformers.modeling_roberta import RobertaLMHead
import pprint
import os
import submitit
import datetime

pp = pprint.PrettyPrinter(indent=2)

model_base_loc = "models/"

model_store = {
    "distilbert": {"mnli": {"orig": "distilbert.mnli/",}},
    "chinese-roberta-large": {"ocnli": {"orig": "roberta.large.ocnli/"}},
}

data_base_loc = "data/"

data_store = {
    "snli_dev": "build/snli/dev.jsonl",
    "snli_test": "build/snli/test.jsonl",
    "mnli_m_dev": "build/mnli/m_dev.jsonl",
    "mnli_mm_dev": "build/mnli/mm_dev.jsonl",
    "anli_r1_dev": "build/anli/r1/dev.jsonl",
    "anli_r1_test": "build/anli/r1/test.jsonl",
    "anli_r2_dev": "build/anli/r2/dev.jsonl",
    "anli_r2_test": "build/anli/r2/test.jsonl",
    "anli_r3_dev": "build/anli/r3/dev.jsonl",
    "anli_r3_test": "build/anli/r3/test.jsonl",
    "ocnli_dev": "build/ocnli/dev.jsonl",
    # Random evaluation files, to be copied before calling this program
    "mnli_m_dev_rand": "build/rand/mnli_m_dev_rand.jsonl",
    "mnli_mm_dev_rand": "build/rand/mnli_mm_dev_rand.jsonl",
    "snli_dev_rand": "build/rand/snli_dev_rand.jsonl",
    "snli_test_rand": "build/rand/snli_test_rand.jsonl",
    "anli_r1_dev_rand": "build/rand/anli_r1_dev_rand.jsonl",
    "anli_r2_dev_rand": "build/rand/anli_r2_dev_rand.jsonl",
    "anli_r3_dev_rand": "build/rand/anli_r3_dev_rand.jsonl",
    "ocnli_dev_rand": "build/rand/ocnli_dev_rand.jsonl",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="If set, we only use CPU.")
    parser.add_argument(
        "--model_class_name",
        type=str,
        help="Set the model class of the experiment.",
        required=True,
    )

    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        help="Set the path to save the prediction.",
        default="",
    )

    parser.add_argument(
        "--output_prediction_path",
        type=str,
        default=None,
        help="Set the path to save the prediction.",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )

    parser.add_argument(
        "--max_length", default=156, type=int, help="Max length of the sequences."
    )

    parser.add_argument(
        "--eval_data", type=str, help="The training data used in the experiments."
    )

    parser.add_argument("--train_data", type=str, help="snli")

    parser.add_argument("--train_mode", type=str, help="orig")

    parser.add_argument(
        "--train_with_lm",
        default=False,
        action="store_true",
        help="Train model with LM",
    )

    parser.add_argument(
        "--flip_sent",
        default=False,
        action="store_true",
        help="Flip the hypothesis and premise",
    )

    parser.add_argument("--slurm", default=False, action="store_true")

    args = parser.parse_args()
    return args


def evaluation(args):

    if args.cpu:
        args.global_rank = -1
    else:
        args.global_rank = 0

    ## Resolve paths
    eval_data = args.eval_data
    if ":" not in args.eval_data:
        args.eval_data = (
            f"{args.eval_data}:{Path(data_base_loc) / data_store[args.eval_data]}"
        )
    args.model_checkpoint_path = (
        Path(model_base_loc)
        / model_store[args.model_class_name][args.train_data][args.train_mode]
        / "model.pt"
    )
    args.output_prediction_path = (
        Path("outputs")
        / args.model_class_name
        / f"{args.train_data}_train_{args.train_mode}_dev_{eval_data}"
    )

    model_checkpoint_path = args.model_checkpoint_path
    num_labels = 3
    # we are doing NLI so we set num_labels = 3, for other task we can change this value.

    max_length = args.max_length

    model_class_item = MODEL_CLASSES[args.model_class_name]
    model_name = model_class_item["model_name"]
    do_lower_case = (
        model_class_item["do_lower_case"]
        if "do_lower_case" in model_class_item
        else False
    )

    tokenizer = model_class_item["tokenizer"].from_pretrained(
        model_name,
        cache_dir=str(config.PRO_ROOT / "trans_cache"),
        do_lower_case=do_lower_case,
    )

    model = model_class_item["sequence_classification"].from_pretrained(
        model_name,
        cache_dir=str(config.PRO_ROOT / "trans_cache"),
        num_labels=num_labels,
    )

    state_dict = torch.load(model_checkpoint_path)
    if args.train_with_lm:
        model.lm_head = RobertaLMHead(model.config)

    model.load_state_dict(state_dict, strict=False)

    padding_token_value = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    padding_segement_value = model_class_item["padding_segement_value"]
    padding_att_value = model_class_item["padding_att_value"]
    left_pad = model_class_item["left_pad"] if "left_pad" in model_class_item else False

    batch_size_per_gpu_eval = args.per_gpu_eval_batch_size

    eval_data_str = args.eval_data
    eval_data_name = []
    eval_data_path = []
    eval_data_list = []

    eval_data_named_path = eval_data_str.split(",")

    for named_path in eval_data_named_path:
        ind = named_path.find(":")
        name = named_path[:ind]
        path = named_path[ind + 1 :]
        if name in registered_path:
            d_list = common.load_jsonl(registered_path[name])
        else:
            d_list = common.load_jsonl(path)
        eval_data_name.append(name)
        eval_data_path.append(path)

        eval_data_list.append(d_list)

    batching_schema = {
        "uid": RawFlintField(),
        "y": LabelFlintField(),
        "input_ids": ArrayIndexFlintField(
            pad_idx=padding_token_value, left_pad=left_pad
        ),
        "token_type_ids": ArrayIndexFlintField(
            pad_idx=padding_segement_value, left_pad=left_pad
        ),
        "attention_mask": ArrayIndexFlintField(
            pad_idx=padding_att_value, left_pad=left_pad
        ),
    }

    if args.flip_sent:
        print("Flipping hypothesis and premise")
        data_transformer = FlippedNLITransform(model_name, tokenizer, max_length)
    else:
        data_transformer = NLITransform(model_name, tokenizer, max_length)
    eval_data_loaders = []
    for eval_d_list in eval_data_list:
        d_dataset, d_sampler, d_dataloader = build_eval_dataset_loader_and_sampler(
            eval_d_list, data_transformer, batching_schema, batch_size_per_gpu_eval
        )
        eval_data_loaders.append(d_dataloader)

    if not args.cpu:
        torch.cuda.set_device(0)
        model.cuda(0)

    r_dict = dict()
    # Eval loop:
    for i in range(len(eval_data_name)):
        cur_eval_data_name = eval_data_name[i]
        cur_eval_data_list = eval_data_list[i]
        cur_eval_dataloader = eval_data_loaders[i]
        # cur_eval_raw_data_list = eval_raw_data_list[i]

        evaluation_dataset(
            args,
            cur_eval_dataloader,
            cur_eval_data_list,
            model,
            r_dict,
            eval_name=cur_eval_data_name,
        )

    # save prediction:
    if args.output_prediction_path is not None:
        cur_results_path = Path(args.output_prediction_path)
        if not cur_results_path.exists():
            cur_results_path.mkdir(parents=True)
        for key, item in r_dict.items():
            common.save_jsonl(item["predictions"], cur_results_path / f"{key}.jsonl")

        # avoid saving too many things
        for key, item in r_dict.items():
            del r_dict[key]["predictions"]
        common.save_json(r_dict, cur_results_path / "results_dict.json", indent=2)

    return r_dict


if __name__ == "__main__":
    args = get_args()
    d = datetime.datetime.today()
    main_exp_type = f"eval_{args.model_class_name}_train_{args.train_data}+{args.train_mode}_eval_{args.eval_data}"
    # logdir = Path.cwd()
    exp_dir = Path(".") / "logs" / f"{d.strftime('%Y-%m-%d')}_{main_exp_type}"
    os.makedirs(exp_dir, exist_ok=True)
    if args.slurm:
        # run by submitit
        submitit_logdir = exp_dir / "submitit_logs"
        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=1440,
            slurm_partition="learnfair",
            gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=10,
            slurm_mem="",
        )
        job = executor.submit(evaluation, args)
        print(f"Submitted job {job.job_id}")
    else:
        evaluation(args)
