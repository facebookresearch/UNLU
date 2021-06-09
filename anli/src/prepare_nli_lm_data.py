# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Prepare perplexity evaluation data for NLI samples
## In one csv file, write the row numbers of the nli samples and the source file
## In another text file, paste the actual sentences
import random
from pathlib import Path
import pandas as pd
import argparse
import sys
from tqdm.auto import tqdm

sys.path.append("/private/home/koustuvs/mlp/nli_gen/anli/src/")
from utils.common import load_jsonl
import config
from nli.training import registered_path
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inp", type=str, default="comma separated registered paths (training)"
    )
    parser.add_argument("--num_rows", type=int, default=1000)
    parser.add_argument("--num_splits", type=int, default=1000)
    parser.add_argument("--outp_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    inp_paths = args.inp.split(",")
    rows = []
    for inp_path in inp_paths:
        inp_lines = load_jsonl(registered_path[inp_path])
        inp_lines = [(inp_path, ri, row) for ri, row in enumerate(inp_lines)]
        rows.extend(inp_lines)
    for i in range(args.num_splits):
        # sample rows
        outp_dir = Path(args.outp_dir) / f'nli_{i}'
        if args.num_rows > 0:
            task_rows = random.sample(rows, args.num_rows)
        outp_dir.mkdir(parents=True, exist_ok=True)
        # save csv
        df_rows = []
        for path, ri, row in task_rows:
            r = copy.deepcopy(row)
            r["path"] = path
            r["index"] = ri
            df_rows.append(r)
        df = pd.DataFrame(df_rows)
        df.to_csv(outp_dir / "nli_data.csv")
        outp = []
        for path, ri, row in task_rows:
            outp.append(row["premise"])
            outp.append(row["hypothesis"])
        outp = list(set(outp))
        with open(outp_dir / "lm_input.txt", "w") as fp:
            for line in outp:
                fp.write(line + "\n")
    print("Done")
