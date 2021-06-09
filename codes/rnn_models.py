# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torchtext import data

# from code.rnn import RecurrentEncoder, Encoder, AttnDecoder, Decoder
import torch
import numpy as np
import argparse
import os
import string
import csv

# from comet_ml import OfflineExperiment
import torch
from torch.autograd import Variable

from infersent_comp.models import NLINet
from torchtext.data import BucketIterator

from codes.models import NLI_Model
from tqdm.auto import tqdm
import yaml
from addict import Dict
import tempfile
from pathlib import Path
import spacy

nlp = spacy.load("zh_core_web_lg")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def get_model_args():
    parser = argparse.ArgumentParser()
    # /////////NLI-Args//////////////
    parser = argparse.ArgumentParser(description="NLI training")
    # paths
    parser.add_argument(
        "--nlipath", type=str, default="ocnli", help="NLI data (mnli or snli)"
    )
    parser.add_argument(
        "--outputdir", type=str, default="rnn_models/", help="Output directory"
    )
    parser.add_argument("--outputmodelname", type=str, default="model.pickle")
    # training
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--dpout_model", type=float, default=0.0, help="encoder dropout"
    )
    parser.add_argument(
        "--dpout_fc", type=float, default=0.0, help="classifier dropout"
    )
    parser.add_argument(
        "--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1"
    )
    parser.add_argument(
        "--lrshrink", type=float, default=5, help="shrink factor for sgd"
    )
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
    parser.add_argument(
        "--max_norm", type=float, default=5.0, help="max norm (grad clipping)"
    )
    # model
    parser.add_argument(
        "--encoder_type", type=str, default="InferSent", help="see list of encoders"
    )
    parser.add_argument(
        "--enc_lstm_dim", type=int, default=200, help="encoder nhid dimension"
    )  # 2048
    parser.add_argument(
        "--n_enc_layers", type=int, default=1, help="encoder num layers"
    )
    parser.add_argument("--fc_dim", type=int, default=200, help="nhid of fc layers")
    parser.add_argument(
        "--n_classes", type=int, default=5, help="entailment/neutral/contradiction"
    )
    parser.add_argument("--pool_type", type=str, default="max", help="max or mean")

    # gpu
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
    parser.add_argument("--seed", type=int, default=100, help="seed")

    # data
    parser.add_argument(
        "--word_emb_dim", type=int, default=300, help="word embedding dimension"
    )
    parser.add_argument(
        "--word_emb_type",
        type=str,
        default="normal",
        help="word embedding type, either glove or normal",
    )

    params, _ = parser.parse_known_args()
    return params


encoder_types = [
    "BiLSTM",
    "InferSent",
    "BLSTMprojEncoder",
    "BGRUlastEncoder",
    "InnerAttentionMILAEncoder",
    "InnerAttentionYANGEncoder",
    "InnerAttentionNAACLEncoder",
    "ConvNetEncoder",
    "LSTMEncoder",
]


def getEmbeddingWeights(vocab, dataset="snli"):
    emb_dict = {}
    with open(f"rnn_models/vocab/glove_" + dataset + "_embeddings.tsv", "r") as f:
        for l in f:
            line = l.split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            emb_dict.update({word: vect})
    vectors = []
    for i in range(len(vocab)):
        vectors += [emb_dict[vocab[i]]]
    return torch.from_numpy(np.stack(vectors)).to(device)


def tokenizer(string_):
    return string_.split()


def remove_non_ascii(string_, vocab=None):
    tokens = string_.split()
    printable = set(string.printable)
    scr_tokens = []
    for w in tokens:
        if vocab is None or w in vocab:
            word = "".join(filter(lambda x: x in printable, w))
            scr_tokens += [word]
    return " ".join(scr_tokens)


train_dialog = None


def evaluate(nli_net, valid_iter, inv_label=None):
    nli_net.eval()
    pred = []
    logits = []
    pb = tqdm(total=len(valid_iter))
    for i, batch in enumerate(valid_iter):
        with torch.no_grad():
            # prepare batch
            s1_batch, s1_len = batch.Sentence1
            s2_batch, s2_len = batch.Sentence2
            s1_batch, s2_batch = (
                Variable(s1_batch.to(device)),
                Variable(s2_batch.to(device)),
            )
            tgt_batch = batch.Label.to(device)
            # model forward
            output, (s1_out, s2_out) = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

            pred += [_.item() for _ in output.data.max(1)[1]]

            logits += [_ for _ in output.cpu()]
        # test_prediction = inv_label[pred[b_index].item()]
        pb.update(1)
    pb.close()
    return pred, logits


def getScores(
    nli_net,
    prem=[],
    hyp=[],
    train_dialog=None,
    word_vec=None,
    dataset="snli",
    target="test.csv",
    batch_size=128,
):
    with tempfile.TemporaryDirectory() as dirpath:
        target = Path(dirpath) / target
        fieldnames = ["ContextID", "Sentence1", "Sentence2", "Label"]

        f = open(target, "w")
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        for i, (p, h) in enumerate(zip(prem, hyp)):
            c_id = i + 1
            if dataset == "ocnli":
                sentence1 = " ".join([str(_) for _ in nlp(p)])
                sentence2 = " ".join([str(_) for _ in nlp(h)])
                prefix = ""
                suffix = ""
            else:
                sentence1 = " ".join(tokenizer(remove_non_ascii(p)))
                sentence2 = " ".join(tokenizer(remove_non_ascii(h)))
                prefix = "processed_"
                suffix = "_glove"
            label = "entailment"
            d = {
                "ContextID": c_id,
                "Sentence1": sentence1,
                "Sentence2": sentence2,
                "Label": label,
            }
            writer.writerow(d)
        f.close()

        # Data Iterator

        TEXT = data.Field(
            fix_length=None,
            eos_token="</s>",
            pad_token="<pad>",
            init_token="<s>",
            unk_token="<unk>",
            include_lengths=True,
        )
        label = data.LabelField()

        train_dialog = data.TabularDataset(
            path=os.path.join(
                "rnn_models", dataset, prefix + dataset + "_train" + suffix + ".csv"
            ),
            format="csv",
            fields=[
                ("context_id", None),
                ("Sentence1", TEXT),
                ("Sentence2", TEXT),
                ("Label", label),
            ],
        )

        test_dialog = data.TabularDataset(
            path=os.path.join(target),
            format="csv",
            fields=[
                ("context_id", None),
                ("Sentence1", TEXT),
                ("Sentence2", TEXT),
                ("Label", label),
            ],
        )

        TEXT.build_vocab(train_dialog)
        label.build_vocab(train_dialog)
        print(len(TEXT.vocab.itos), "--------------------------")
        return train_dialog, test_dialog, TEXT, label


def get_model_config(params):
    map_classes = {
        "ocnli": 3,
        "scitail": 2,
        "snli": 5,
        "mnli": 3,
        "mnli_rand": 3,
        "anli": 3,
        "fever": 3,
    }
    n_words_map = {"ocnli": 26722, "snli": 25360, "mnli": 67814, "mnli_rand": 59177}
    config = {
        "n_words": n_words_map[params.nlipath],
        "word_emb_dim": params.word_emb_dim,
        "enc_lstm_dim": params.enc_lstm_dim,
        "n_enc_layers": params.n_enc_layers,
        "dpout_model": params.dpout_model,
        "dpout_fc": params.dpout_fc,
        "fc_dim": params.fc_dim,
        "bsize": params.batch_size,
        "n_classes": map_classes[params.nlipath],
        "pool_type": params.pool_type,
        "nonlinear_fc": params.nonlinear_fc,
        "encoder_type": params.encoder_type,
        "use_cuda": False,
    }
    return config


class InferSent(NLI_Model):
    def __init__(self, args):
        super().__init__(args)
        ## Model specific init
        params = get_model_args()
        params.encoder_type = args.encoder_type
        nlipath = "mnli" if "nlipath" not in args else args.nlipath
        params.nlipath = "ocnli" if "ocnli" in args.model_name else nlipath
        if "outputdir" in args:
            print(f"Changing otuputdir to {args.outputdir}")
            params.outputdir = args.outputdir
        print(f"NLIPATH : {params.nlipath}")
        self.params = params
        exp_folder = os.path.join(
            params.outputdir,
            params.nlipath,
            params.encoder_type,
            "exp_seed_{}".format(params.seed),
        )
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        # set proper name
        save_folder_name = os.path.join(exp_folder, "model")

        params.outputmodelname = os.path.join(
            save_folder_name, "{}_model.pkl".format(params.encoder_type)
        )

        if params.nlipath != "ocnli":
            word_vec = getEmbeddingWeights(
                [
                    _.split("\n")[0]
                    for _ in open(
                        "rnn_models/vocab/glove_" + params.nlipath + "_vocab.txt"
                    ).readlines()
                ],
                params.nlipath,
            )
        else:
            word_vec = None
        model_config = get_model_config(params)
        print(
            params.nlipath,
            params.outputmodelname,
            word_vec is None,
            "-----------!!------------",
        )
        self.model = NLINet(model_config, weights=word_vec)
        self.model.load_state_dict(
            torch.load(params.outputmodelname, map_location=torch.device("cpu"))
        )
        self.model.eval()
        # nothing to do here, as its handled internally
        self.label_fn = {"c": "c", "n": "n", "e": "e"}

    def prepare_batches(
        self,
        data,
        batch_size,
        sent1_label="sentence1",
        sent2_label="sentence2",
        target_label="label",
        **kwargs,
    ):
        print("preparing data ...")
        _, test_dialog, _, label = getScores(
            self.model,
            [row["premise"] for row in data],
            [row["hypothesis"] for row in data],
            dataset=self.params.nlipath,
        )
        print("done, now predicting ..")
        test_dialog_iter = BucketIterator(
            test_dialog,
            batch_size=batch_size,
            sort_key=lambda x: len(x.Label),
            sort_within_batch=False,
            shuffle=False,
            device=device,
        )
        meta = [
            [row["uid"], row[sent1_label], row[sent2_label], row[target_label]]
            for row in data
        ]
        return [[[test_dialog_iter, label], meta]]

    def predict_batch(self, inp):
        test_dialog_iter, label = inp
        pred, logits = evaluate(self.model, test_dialog_iter, label.vocab.itos)
        pred = [label.vocab.itos[pred[_]][0] for _ in range(len(pred))]
        logits = [l.tolist() for l in logits]
        return pred, logits
        # outp = [
        #     {"predicted_label": pred[i][0], "logits": logits[i].tolist()}
        #     for i in range(len(pred))
        # ]
        # return outp


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

    model = InferSent(Dict(args))
