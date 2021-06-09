# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
# from transformers import RobertaTokenizer
import os
import json
import math
import random
import csv
import string

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def tokenizer(string_):
    return string_.split()


def myround(x, base=20):
    return str(int(base * max(1, math.ceil(x / base)) / 20))


def remove_non_ascii(string_, vocab=None):
    tokens = string_.split()
    printable = set(string.printable)
    scr_tokens = []
    for w in tokens:
        if vocab is None or w in vocab:
            word = "".join(filter(lambda x: x in printable, w))
            scr_tokens += [word]
    return " ".join(scr_tokens)


def getID(lettersCount=4, digitsCount=3):
    sampleStr = "".join(
        (random.choice(string.ascii_letters) for i in range(lettersCount))
    )
    sampleStr += "".join((random.choice(string.digits) for i in range(digitsCount)))

    # Convert string to list and shuffle it to mix letters and digits
    sampleList = list(sampleStr)
    random.shuffle(sampleList)
    finalString = "".join(sampleList)
    return finalString


def createCSV(dataset, type):
    print(type)
    con_num = 0
    fieldnames = ["ContextID", "Sentence1", "Sentence2", "Label"]
    if type == "train":
        target1 = open("processed_mnli_" + type + "_glove.csv", "w")
        writer1 = csv.DictWriter(target1, fieldnames=fieldnames)
        target2 = open("processed_mnli_dev_glove.csv", "w")
        writer2 = csv.DictWriter(target2, fieldnames=fieldnames)
    else:
        target1 = open("processed_mnli_test_glove.csv", "w")
        writer1 = csv.DictWriter(target1, fieldnames=fieldnames)
    # writer.writerow(dict(zip(fieldnames, fieldnames)))
    clean_vocab = open(
        os.path.join("..", "..", "utils", "glove_mnli_vocab.txt")
    ).readlines()
    clean_vocab = [w.strip() for w in clean_vocab]
    cnt = False
    for row in dataset:
        c_id = getID()
        con_num += 1
        if cnt:
            sentence1 = " ".join(tokenizer(remove_non_ascii(row[5], clean_vocab)))
            sentence2 = " ".join(tokenizer(remove_non_ascii(row[6], clean_vocab)))
            label = row[0]
            d = {
                "ContextID": c_id,
                "Sentence1": sentence1,
                "Sentence2": sentence2,
                "Label": label,
            }
            if type == "train":
                if con_num > 382703:
                    writer2.writerow(d)
                else:
                    writer1.writerow(d)
            else:
                writer1.writerow(d)
        cnt = True
    if type == "train":
        target1.close()
        target2.close()
    else:
        target1.close()


def get_dataset(data_folder="."):
    if data_folder == "dev":
        data_file = os.path.join(
            "multinli_1.0", "multinli_1.0_" + data_folder + "_matched.txt"
        )
    elif data_folder == "train":
        data_file = os.path.join("multinli_1.0", "multinli_1.0_" + data_folder + ".txt")
    Data = csv.reader(open(data_file), delimiter="\t", quoting=csv.QUOTE_NONE)
    next(Data)
    createCSV(Data, data_folder)

    # logger.info("Tokenize and encode the dataset")


if __name__ == "__main__":
    get_dataset("dev")
    get_dataset("train")
