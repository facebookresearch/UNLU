# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
from torchtext import data

# from code.rnn import RecurrentEncoder, Encoder, AttnDecoder, Decoder
from dataset_utils.data_iterator import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from utils.eval_metric import getBLEU
from utils.optim import GradualWarmupScheduler
from utils.transformer_utils import create_masks
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import logging
import random
import itertools
import math
import csv
import sys
from filelock import FileLock
import ray

import time

# from comet_ml import OfflineExperiment
import torch
from torch.autograd import Variable
import torch.nn as nn

from infersent_comp.data import get_nli, get_batch, build_vocab, DICO_LABEL
from infersent_comp.mutils import get_optimizer
from infersent_comp.models import NLINet
import pandas as pd
from collections import Counter
import pdb

ray.init(num_gpus=1)
# commandline arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getEmbeddingWeights(vocab, dataset="snli"):
    emb_dict = {}
    with open(f"utils/glove_" + dataset + "_embeddings.tsv", "r") as f:
        for l in f:
            line = l.split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            emb_dict.update({word: vect})
    vectors = []
    for i in range(len(vocab)):
        vectors += [emb_dict[vocab[i]]]
    return torch.from_numpy(np.stack(vectors)).to(device)


def trainepoch(nli_net, train_iter, optimizer, loss_fn, epoch, params):
    print("\nTRAINING : Epoch " + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.0
    # shuffle the data

    optimizer.param_groups[0]["lr"] = (
        optimizer.param_groups[0]["lr"] * params.decay
        if epoch > 1 and "sgd" in params.optimizer
        else optimizer.param_groups[0]["lr"]
    )
    print("Learning rate : {0}".format(optimizer.param_groups[0]["lr"]))
    total_samples = 0

    for i, batch in enumerate(train_iter):
        # prepare batch
        s1_batch, s1_len = batch.Sentence1
        s2_batch, s2_len = batch.Sentence2
        s1_batch, s2_batch = (
            Variable(s1_batch.to(device)),
            Variable(s2_batch.to(device)),
        )
        tgt_batch = batch.Label.to(device)
        k = s1_batch.size(1)  # actual batch size
        total_samples += k
        # model forward
        output, (s1_out, s2_out) = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        # loss
        # pdb.set_trace()
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())
        words_count += s1_batch.nelement() + s2_batch.nelement()

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0

        for p in nli_net.parameters():
            if p.requires_grad:
                p.grad.div_(k)  # divide by the actual batch size
                total_norm += p.grad.norm() ** 2
        total_norm = np.sqrt(total_norm.item())

        if total_norm > params.max_norm:
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0][
            "lr"
        ]  # current lr (no external "lr", for adam)
        optimizer.param_groups[0]["lr"] = current_lr * shrink_factor  # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]["lr"] = current_lr

        if len(all_costs) == 100:
            logs.append(
                "{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}".format(
                    (i) * params.batch_size,
                    round(np.mean(all_costs), 2),
                    int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                    int(words_count * 1.0 / (time.time() - last_time)),
                    round(100.0 * correct / ((i + 1) * params.batch_size), 2),
                )
            )
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct / total_samples, 2)
    print("results : epoch {0} ; mean accuracy train : {1}".format(epoch, train_acc))
    # ex.log_metric('train_accuracy', train_acc, step=epoch)
    return train_acc


def evaluate(
    nli_net,
    valid_iter,
    optimizer,
    epoch,
    train_config,
    params,
    eval_type="valid",
    test_folder=None,
    inv_label=None,
    itos_vocab=None,
    final_eval=False,
    probe_set="0",
):
    nli_net.eval()
    correct = 0.0
    test_prediction = []
    s1 = []
    s2 = []
    target = []

    if eval_type == "valid":
        print("\nVALIDATION : Epoch {0}".format(epoch))
    total_samples = 0
    for i, batch in enumerate(valid_iter):
        # prepare batch
        s1_batch, s1_len = batch.Sentence1
        s2_batch, s2_len = batch.Sentence2
        s1_batch, s2_batch = (
            Variable(s1_batch.to(device)),
            Variable(s2_batch.to(device)),
        )
        tgt_batch = batch.Label.to(device)
        total_samples += s1_batch.size(1)

        # model forward
        output, (s1_out, s2_out) = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        if eval_type == "test":
            for b_index in range(len(batch)):
                test_prediction = inv_label[pred[b_index].item()]
                s1 = " ".join(
                    [
                        itos_vocab[idx.item()]
                        for idx in batch.Sentence1[0][
                            : batch.Sentence1[1][b_index], b_index
                        ]
                    ]
                ).replace("Ġ", " ")
                s2 = " ".join(
                    [
                        itos_vocab[idx.item()]
                        for idx in batch.Sentence2[0][
                            : batch.Sentence2[1][b_index], b_index
                        ]
                    ]
                ).replace("Ġ", " ")
                target = inv_label[batch.Label[b_index]]
                res_file = os.path.join(test_folder, probe_set + "samples.txt")
                lock = FileLock(
                    os.path.join(test_folder, probe_set + "_samples.txt.new.lock")
                )
                with lock:
                    with open(res_file, "a") as f:
                        f.write(
                            "S1: "
                            + s1
                            + "\n"
                            + "S2: "
                            + s2
                            + "\n"
                            + "Target: "
                            + target
                            + "\n"
                            + "Predicted: "
                            + test_prediction
                            + "\n\n"
                        )
                    lock.release()

    # save model
    eval_acc = round(100 * correct / total_samples, 2)
    if final_eval:
        print("finalgrep : accuracy {0} : {1}".format(eval_type, eval_acc))
        # ex.log_metric('{}_accuracy'.format(eval_type), eval_acc, step=epoch)
    else:
        print(
            "togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}".format(
                epoch, eval_type, eval_acc
            )
        )
        # ex.log_metric('{}_accuracy'.format(eval_type), eval_acc, step=epoch)

    if eval_type == "valid" and epoch <= params.n_epochs:
        if eval_acc > train_config["val_acc_best"]:
            print("saving model at epoch {0}".format(epoch))
            # if not os.path.exists(params.outputdir):
            #    os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), params.outputmodelname)
            train_config["val_acc_best"] = eval_acc
        else:
            if "sgd" in params.optimizer:
                optimizer.param_groups[0]["lr"] = (
                    optimizer.param_groups[0]["lr"] / params.lrshrink
                )
                print(
                    "Shrinking lr by : {0}. New lr = {1}".format(
                        params.lrshrink, optimizer.param_groups[0]["lr"]
                    )
                )
                if optimizer.param_groups[0]["lr"] < params.minlr:
                    train_config["stop_training"] = True
            if "adam" in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                train_config["stop_training"] = train_config["adam_stop"]
                # adam_stop = True

    return eval_acc, optimizer, train_config


@ray.remote(num_gpus=1)
def HyperEvaluate(config):
    print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-ip-address=")  # ,192.168.2.19
    parser.add_argument("--node-manager-port=")
    parser.add_argument("--object-store-name=")
    parser.add_argument(
        "--raylet-name="
    )  # /tmp/ray/session_2020-07-15_12-00-45_292745_38156/sockets/raylet
    parser.add_argument("--redis-address=")  # 192.168.2.19:6379
    parser.add_argument("--config-list=", action="store_true")  #
    parser.add_argument("--temp-dir=")  # /tmp/ray
    parser.add_argument("--redis-password=")  # 5241590000000000
    # /////////NLI-Args//////////////
    parser = argparse.ArgumentParser(description="NLI training")
    # paths
    parser.add_argument(
        "--nlipath",
        type=str,
        default=config["dataset"],
        help="NLI data (SNLI or MultiNLI)",
    )
    parser.add_argument(
        "--outputdir", type=str, default="savedir_test/", help="Output directory"
    )
    parser.add_argument("--outputmodelname", type=str, default="model.pickle")
    parser.add_argument(
        "--word_emb_path",
        type=str,
        default="dataset/GloVe/glove.840B.300d.txt",
        help="word embedding file path",
    )

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
    parser.add_argument(
        "--probe_task_id",
        type=str,
        default=config["task_id"],
        help="task index for the probe tasks",
    )
    # model
    parser.add_argument(
        "--encoder_type",
        type=str,
        default=config["encoder_type"],
        help="see list of encoders",
    )
    parser.add_argument(
        "--enc_lstm_dim", type=int, default=200, help="encoder nhid dimension"
    )  # 2048
    parser.add_argument(
        "--n_enc_layers", type=int, default=1, help="encoder num layers"
    )
    parser.add_argument("--fc_dim", type=int, default=200, help="nhid of fc layers")
    parser.add_argument(
        "--n_classes",
        type=int,
        default=config["num_classes"],
        help="entailment/neutral/contradiction",
    )
    parser.add_argument("--pool_type", type=str, default="max", help="max or mean")

    # gpu
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
    parser.add_argument("--seed", type=int, default=config["seed"], help="seed")

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

    # comet
    # parser.add_argument("--comet_apikey", type=str, default='', help="comet api key")
    # parser.add_argument("--comet_workspace", type=str, default='', help="comet workspace")
    # parser.add_argument("--comet_project", type=str, default='', help="comet project name")
    # parser.add_argument("--comet_disabled", action='store_true', help="if true, disable comet")

    params, _ = parser.parse_known_args()
    print("Came here")
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
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)

    test_sample_folder = os.path.join(exp_folder, "samples_test")
    if not os.path.exists(test_sample_folder):
        os.makedirs(test_sample_folder)
    params.outputmodelname = os.path.join(
        save_folder_name, "{}_model.pkl".format(params.encoder_type)
    )
    # print parameters passed, and all parameters
    print("\ntogrep : {0}\n".format(sys.argv[1:]))
    print(params)
    pr = vars(params)

    # ex = OfflineExperiment(
    #                 workspace=pr['comet_workspace'],
    #                 project_name=pr['comet_project'],
    #                 disabled=pr['comet_disabled'],
    #                 offline_directory= os.path.join(save_folder_name,'comet_runs'))
    #
    # ex.log_parameters(pr)
    # ex.set_name(pr['encoder_type'])

    """
    SEED
    """
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # torch.cuda.manual_seed(params.seed)

    """
    DATA
    """
    train, valid, test, vocab, label_vocab = DataIterator_Gloveprobe(
        batch_size=params.batch_size,
        dataset=params.nlipath,
        max_length=20,
        probe_set=params.probe_task_id,
        prefix="processed_",
    )
    # word_vec = build_vocab(train['s1'] + train['s2'] +
    #                        valid['s1'] + valid['s2'] +
    #                        test['s1'] + test['s2'], params.word_emb_path, emb_dim=params.word_emb_dim,
    #                        wtype=params.word_emb_type)

    # for split in ['s1', 's2']:
    #     for data_type in ['train', 'valid', 'test']:
    #         eval(data_type)[split] = np.array([['<s>'] +
    #             [word for word in sent.split() if word in word_vec] +
    #             ['</s>'] for sent in eval(data_type)[split]])

    # Train label class balancing
    weights = [2, 2, 2, 0.3, 7, 2, 6]
    # invert the weights by values
    word_vec = getEmbeddingWeights(vocab.itos, params.nlipath)
    print("Embeddings loaded")
    """
    MODEL
    """
    # model config
    config_nli_model = {
        "n_words": len(vocab),
        "word_emb_dim": params.word_emb_dim,
        "enc_lstm_dim": params.enc_lstm_dim,
        "n_enc_layers": params.n_enc_layers,
        "dpout_model": params.dpout_model,
        "dpout_fc": params.dpout_fc,
        "fc_dim": params.fc_dim,
        "bsize": params.batch_size,
        "n_classes": params.n_classes,
        "pool_type": params.pool_type,
        "nonlinear_fc": params.nonlinear_fc,
        "encoder_type": params.encoder_type,
        "use_cuda": True,
    }

    # model
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
    assert params.encoder_type in encoder_types, "encoder_type must be in " + str(
        encoder_types
    )
    nli_net = NLINet(config_nli_model, weights=word_vec)
    print(nli_net)

    weight = torch.FloatTensor(weights)
    loss_fn = nn.CrossEntropyLoss(weight=weight[: params.n_classes])
    loss_fn.size_average = False

    # optimizer
    optim_fn, optim_params = get_optimizer(params.optimizer)
    optimizer = optim_fn(nli_net.parameters(), **optim_params)

    # cuda by default
    nli_net.to(device)
    loss_fn.to(device)

    """
    TRAIN
    """
    train_config = {
        "val_acc_best": -1e10,
        "adam_stop": False,
        "stop_training": False,
        "lr": optim_params["lr"] if "sgd" in params.optimizer else None,
    }

    """
    Train model on Natural Language Inference task
    """
    epoch = 0

    # Run best model on test set.
    nli_net.load_state_dict(torch.load(params.outputmodelname))

    print("\nTEST : Epoch {0}".format(epoch))
    test_acc, _, _ = evaluate(
        nli_net,
        test,
        optimizer,
        epoch,
        train_config,
        params,
        eval_type="test",
        test_folder=test_sample_folder,
        inv_label=label_vocab.itos,
        itos_vocab=vocab.itos,
        final_eval=True,
        probe_set=params.probe_task_id,
    )
    lock = FileLock(os.path.join(save_folder_name, "logs.txt" + ".new.lock"))
    with lock:
        with open(
            os.path.join(save_folder_name, "logs_" + params.probe_task_id + ".txt"), "a"
        ) as f:
            f.write(f"| Epoch: {epoch:03} | Test Acc: {test_acc:.3f} \n")
        lock.release()
    # ex.log_asset(file_name=res_file, file_like_object=open(res_file, 'r'))

    # Save word vectors

    return test_acc


# Test set performance
t_encoders = ["BiLSTM"]  # [ 'BLSTMprojEncoder', 'BGRUlastEncoder',
#    'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
#   'InnerAttentionNAACLEncoder', 'LSTMEncoder', 'InferSent', 'ConvNetEncoder']
t_seeds = [9999]
t_dataset = ["mnli"]  # ,'snli']#, 'personachat', 'dailydialog']
num_classes_ = {"scitail": 2, "snli": 5, "mnli": 3, "anli": 3, "fever": 3}
t_tasks = [0]  # 0,1,2,3,4,5]
best_hyperparameters = None
best_accuracy = 0
# A list holding the object IDs for all of the experiments that we have
# launched but have not yet been processed.
remaining_ids = []
# A dictionary mapping an experiment's object ID to its hyperparameters.
# hyerparameters used for that experiment.
hyperparameters_mapping = {}

for s, d, m, t in itertools.product(t_seeds, t_dataset, t_encoders, t_tasks):
    config = {}
    config["encoder_type"] = m
    config["seed"] = s
    config["dataset"] = d
    config["task_id"] = str(t)
    config["num_classes"] = num_classes_[d]
    accuracy_id = HyperEvaluate.remote(config)
    remaining_ids.append(accuracy_id)
    hyperparameters_mapping[accuracy_id] = config

###########################################################################
# Process each hyperparameter and corresponding accuracy in the order that
# they finish to store the hyperparameters with the best accuracy.

# Fetch and print the results of the tasks in the order that they complete.

while remaining_ids:
    # Use ray.wait to get the object ID of the first task that completes.
    done_ids, remaining_ids = ray.wait(remaining_ids)
    # There is only one return result by default.
    result_id = done_ids[0]

    hyperparameters = hyperparameters_mapping[result_id]
    accuracy = ray.get(result_id)
    print(
        """We achieve {:7.2f}% BLEU in {} with:
        seed: {}
        model: {}
      """.format(
            accuracy,
            hyperparameters["dataset"],
            hyperparameters["seed"],
            hyperparameters["encoder_type"],
        )
    )
