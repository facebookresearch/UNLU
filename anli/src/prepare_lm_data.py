# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Prepare data for Language modelling
import argparse
from utils.common import load_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default="")
    parser.add_argument("--outp", type=str, default="")
    args = parser.parse_args()
    inp_file = load_jsonl(args.inp)
    outp = []
    for row in inp_file:
        outp.append(row["premise"].rstrip())
        outp.append(row["hypothesis"].rstrip())

    outp = list(set(outp))
    with open(args.outp, "w") as fp:
        for line in outp:
            fp.write(line + "\n")
    print("Done")
