# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.

## Call random eval scripts
import argparse
import yaml
from codes.random_eval import main_eval
from addict import Dict
import submitit
from pathlib import Path
from datetime import datetime


def run(args):
    main_eval(Dict(args))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_data", default="mnli_m_dev", type=str, help="eval data")
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="location of config file"
    )
    parser.add_argument("--model_type", default="hub", type=str, help="hub/fairseq/hf")
    parser.add_argument(
        "--model_name",
        default="roberta.large.mnli",
        type=str,
        help="appropriate model name",
    )
    parser.add_argument("--keep_order", default=0.0, type=float, help="keep order")
    parser.add_argument("--slurm", action="store_true", default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config = yaml.load(open(args.config))
    config["eval_data"] = args.eval_data
    config["model_type"] = args.model_type
    config["data_prep_config"]["keep_order"] = args.keep_order
    config[args.model_type]["model_name"] = args.model_name
    if args.slurm:
        # run by submitit
        d = datetime.today()
        exp_dir = (
            Path("/checkpoint/acls")
            / "projects"
            / "nli_gen"
            / f"{d.strftime('%Y-%m-%d')}_rand_eval_{args.eval_data}_{args.model_type}_{args.keep_order}"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        submitit_logdir = exp_dir / "submitit_logs"
        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=720,
            slurm_partition="",
            gpus_per_node=1,
            tasks_per_node=1,
            cpus_per_task=10,
            slurm_mem="",
        )
        job = executor.submit(run, config)
        print(f"Submitted job {job.job_id}")
    else:
        run(config)
