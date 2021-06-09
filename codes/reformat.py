# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Copy over data and reformat
import argparse
import sys

sys.path.append("nli_gen")
sys.path.append("nli_gen/anli/")
sys.path.append("nli_gen/anli/src")
from anli.src.utils import common

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_loc", type=str, default="")
    parser.add_argument("--rand_loc", type=str, default="")
    parser.add_argument("--model_orig_loc", type=str, default="")
    parser.add_argument("--model_rand_loc", type=str, default="")
    parser.add_argument("--model_orig_outp", type=str, default="")
    parser.add_argument("--model_rand_outp", type=str, default="")
    args = parser.parse_args()

    orig_outp = common.load_jsonl(args.orig_loc)
    rand_outp = common.load_jsonl(args.rand_loc)
    model_orig_outp = common.load_jsonl(args.model_orig_loc)
    model_rand_outp = common.load_jsonl(args.model_rand_loc)

    # check rows
    for ri, row in enumerate(rand_outp):
        uid = row["uid"].split("_seed")[0]
        try:
            assert orig_outp[int(ri // 100)]["uid"] == uid
        except:
            print(uid, ri)
            break

    # change model orig outp
    ct = 0
    for bi, b in enumerate(orig_outp):
        assert b["uid"] == model_orig_outp[ct]["uid"]
        model_orig_outp[ct]["uid"] = str(model_orig_outp[ct]["uid"])
        model_orig_outp[ct]["premise"] = b["premise"]
        model_orig_outp[ct]["hypothesis"] = b["hypothesis"]
        model_orig_outp[ct]["orig_label"] = b["label"]
        # model_outp[ct]['logits'] = [model_outp[ct]['logits'][-1], orig_outp[ct]['logits'][1], orig_outp[ct]['logits'][0]]
        model_orig_outp[ct]["is_correct"] = (
            model_orig_outp[ct]["orig_label"] == model_orig_outp[ct]["predicted_label"]
        )
        ct += 1

    # change model rand outp
    ct = 0
    for bi, b in enumerate(rand_outp):
        assert b["uid"] == model_rand_outp[ct]["uid"]
        model_rand_outp[ct]["premise"] = b["premise"]
        model_rand_outp[ct]["hypothesis"] = b["hypothesis"]
        model_rand_outp[ct]["orig_label"] = b["label"]
        # model_outp[ct]['logits'] = [model_outp[ct]['logits'][-1], orig_outp[ct]['logits'][1], orig_outp[ct]['logits'][0]]
        model_rand_outp[ct]["is_correct"] = (
            model_rand_outp[ct]["orig_label"] == model_rand_outp[ct]["predicted_label"]
        )
        ct += 1

    # save
    common.save_jsonl(model_orig_outp, args.model_orig_outp)
    common.save_jsonl(model_rand_outp, args.model_rand_outp)
    print("Done")
