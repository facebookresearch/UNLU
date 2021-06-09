# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
import os
import json
import math
import random
import csv
import string
import spacy

nlp = spacy.load("zh_core_web_lg")


def createCSV(dataset, type):
    print(type)
    fieldnames = ["ContextID", "Sentence1", "Sentence2", "Label"]
    target = open(os.path.join(".", "ocnli_" + type + ".csv"), "w")
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    # writer.writerow(dict(zip(fieldnames, fieldnames)))
    label_key = {"c": "contradiction", "e": "entailment", "n": "neutral"}
    ct = 0
    for row in dataset:
        row_dict = json.loads(row)
        c_id = row_dict["id"]
        sentence1 = " ".join([str(_) for _ in nlp(row_dict["sentence1"])])
        sentence2 = " ".join([str(_) for _ in nlp(row_dict["sentence2"])])
        label_pre = row_dict["label"][0]
        if label_pre not in label_key:
            ct += 1
            continue
        label = label_key[label_pre]
        d = {
            "ContextID": c_id,
            "Sentence1": sentence1,
            "Sentence2": sentence2,
            "Label": label,
        }
        writer.writerow(d)
    target.close()
    print(f"Skipped : {ct} due to no unambiguious labels")


def get_dataset(data_folder="."):
    data_file = os.path.join(".", data_folder + ".json")
    Data = open(data_file).readlines()  # csv.reader(open(data_file),delimiter='\t')
    createCSV(Data, data_folder)

    # logger.info("Tokenize and encode the dataset")


if __name__ == "__main__":
    get_dataset("dev")
    get_dataset("train")
