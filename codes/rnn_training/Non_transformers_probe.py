# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
from torchtext import data
#from code.rnn import RecurrentEncoder, Encoder, AttnDecoder, Decoder
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
import string
import itertools
import math
import csv
import sys
from filelock import FileLock
import ray

import time

#from comet_ml import OfflineExperiment
import torch
from torch.autograd import Variable
import torch.nn as nn

from infersent_comp.data import get_nli, get_batch, build_vocab, DICO_LABEL
from infersent_comp.mutils import get_optimizer
from infersent_comp.models import NLINet
import pandas as pd
from collections import Counter
from torchtext.data import BucketIterator
import pdb

ray.init(num_gpus=3)
# commandline arguments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getEmbeddingWeights(vocab,dataset='snli'):
    emb_dict = {}
    with open(f'utils/glove_'+dataset+'_embeddings.tsv', 'r') as f:
        for l in f:
            line = l.split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            emb_dict.update({word:vect})
    vectors = []
    for i in range(len(vocab)):
        vectors+=[emb_dict[vocab[i]]]
    return torch.from_numpy(np.stack(vectors)).to(device)

def tokenizer(string_):
    return string_.split()

def remove_non_ascii(string_,vocab = None):
    tokens = string_.split()
    printable = set(string.printable)
    scr_tokens = []
    for w in tokens:
        if vocab is None or w in vocab:
            word = ''.join(filter(lambda x: x in printable, w))
            scr_tokens += [word]
    return ' '.join(scr_tokens)

def evaluate(nli_net = None, valid_iter = None, inv_label = None, itos_vocab = None, samples_file = None):
    nli_net.eval()
    pred = []
    logits = []
    for i, batch in enumerate(valid_iter):
        # prepare batch
        s1_batch, s1_len = batch.Sentence1
        s2_batch, s2_len = batch.Sentence2
        s1_batch, s2_batch = Variable(s1_batch.to(device)), Variable(s2_batch.to(device))
        tgt_batch = batch.Label.to(device)
        # model forward
        output, (s1_out, s2_out) = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = [_.item() for _ in output.data.max(1)[1]]

        logits = [_ for _ in output.cpu()]
        #test_prediction = inv_label[pred[b_index].item()]
        for b_index in range(len(batch)):
            uid = batch.ContextID[b_index]
            test_prediction = inv_label[pred[b_index]]
            s1 = ' '.join([itos_vocab[idx.item()] for idx in batch.Sentence1[0][:batch.Sentence1[1][b_index],b_index]]).replace('Ġ',' ')
            s2 = ' '.join([itos_vocab[idx.item()] for idx in batch.Sentence2[0][:batch.Sentence2[1][b_index],b_index]]).replace('Ġ',' ')
            target = inv_label[batch.Label[b_index]]
            logit = [_.item() for _ in output[b_index]]
            is_correct = True if target == test_prediction else False
            lock = FileLock(samples_file+'.lock')
            with lock:
                with open(samples_file,'a') as f:
                    f.write('{ uid:' uid + ', premise: '+s2 +', hypothesis: '+ s1 +', orig_label: ' + target +', model: '+test_prediction +', is_correct:' + str(is_correct) + ', logits:' + str(logit)+'}\n')
                lock.release()
    return pred, logits

def getScores(nli_net= None, dev = 'file.csv', train_dialog = None, word_vec = None, samples_file = 'some.jsonl', dataset = 'mnli', batch_size = 128):

    fieldnames=['ContextID','Sentence1', 'Sentence2', 'Label']

    # Data Iterator

    TEXT = data.Field(fix_length=None, eos_token = '</s>', pad_token = '<pad>', init_token = '<s>', unk_token = '<unk>', include_lengths=True)
    label = data.LabelField()

    train_dialog = data.TabularDataset(path=os.path.join('datasets',dataset,'processed_'+dataset+'_train_glove.csv'), \
    format='csv',fields=[('context_id',None),('Sentence1',TEXT),('Sentence2',TEXT),('Label',label)])


    test_dialog =  data.TabularDataset(path=dev, \
    format='csv',fields=[('context_id',None),('Sentence1',TEXT),('Sentence2',TEXT),('Label',label)])

    TEXT.build_vocab(train_dialog)
    label.build_vocab(train_dialog)


    _, test_dialog_iter  = BucketIterator.splits((train_dialog, test_dialog),\
    batch_size = batch_size , sort_key=lambda x: len(x.Label), sort_within_batch = False, device = device)

    evaluate(nli_net, test_dialog_iter, label.vocab.itos, TEXT.vocab.itos, samples_file)

    return pred, logits, label.vocab.itos
@ray.remote(num_gpus=1)
def HyperEvaluate(config):
    print(config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-ip-address=')#,192.168.2.19
    parser.add_argument('--node-manager-port=')
    parser.add_argument('--object-store-name=')
    parser.add_argument('--raylet-name=')#/tmp/ray/session_2020-07-15_12-00-45_292745_38156/sockets/raylet
    parser.add_argument('--redis-address=')#192.168.2.19:6379
    parser.add_argument('--config-list=',action='store_true')#
    parser.add_argument('--temp-dir=')#/tmp/ray
    parser.add_argument('--redis-password=')#5241590000000000
    #/////////NLI-Args//////////////
    parser = argparse.ArgumentParser(description='NLI training')
    # paths
    parser.add_argument("--nlipath", type=str, default=config['dataset'], help="NLI data (SNLI or MultiNLI)")
    parser.add_argument("--outputdir", type=str, default='savedir_van/', help="Output directory")
    parser.add_argument("--outputmodelname", type=str, default='model.pickle')
    parser.add_argument("--word_emb_path", type=str, default="dataset/GloVe/glove.840B.300d.txt", help="word embedding file path")

    # training
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dpout_model", type=float, default=0.2, help="encoder dropout")
    parser.add_argument("--dpout_fc", type=float, default=0.2, help="classifier dropout")
    parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.001", help="adam or sgd,lr=0.1")
    parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
    parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
    parser.add_argument("--minlr", type=float, default=1e-10, help="minimum lr")
    parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

    # model
    parser.add_argument("--encoder_type", type=str, default=config['encoder_type'], help="see list of encoders")
    parser.add_argument("--enc_lstm_dim", type=int, default=200, help="encoder nhid dimension")#2048
    parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
    parser.add_argument("--fc_dim", type=int, default=200, help="nhid of fc layers")
    parser.add_argument("--n_classes", type=int, default=config['num_classes'], help="entailment/neutral/contradiction")
    parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

    # gpu
    parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
    parser.add_argument("--seed", type=int, default=config['seed'], help="seed")

    # data
    parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")
    parser.add_argument("--word_emb_type", type=str, default='normal', help="word embedding type, either glove or normal")

    # comet
    # parser.add_argument("--comet_apikey", type=str, default='', help="comet api key")
    # parser.add_argument("--comet_workspace", type=str, default='', help="comet workspace")
    # parser.add_argument("--comet_project", type=str, default='', help="comet project name")
    # parser.add_argument("--comet_disabled", action='store_true', help="if true, disable comet")

    params, _ = parser.parse_known_args()

    print('Came here')
    exp_folder = os.path.join(params.outputdir, params.nlipath, params.encoder_type,'exp_seed_{}'.format(params.seed))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    # set proper name
    save_folder_name = os.path.join(exp_folder, 'model')
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)

    test_sample_folder = os.path.join(exp_folder,'samples_test')
    if not os.path.exists(test_sample_folder):
        os.makedirs(test_sample_folder)
    params.outputmodelname = os.path.join(save_folder_name, '{}_model.pkl'.format(params.encoder_type))
    # print parameters passed, and all parameters
    print('\ntogrep : {0}\n'.format(sys.argv[1:]))
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
#    np.random.seed(params.seed)
#    torch.manual_seed(params.seed)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # torch.cuda.manual_seed(params.seed)
    weights = [2, 2, 2, 0.3, 7, 2, 6]
    word_vec = getEmbeddingWeights([_.split('\n')[0] for _ in open('utils/glove_'+params.nlipath+'_vocab.txt').readlines()], params.nlipath)
    n_words_map = {'snli': 25360, 'mnli':67814}
    config_nli_model = {
        'n_words'        :  n_words_map[params.nlipath]          ,
        'word_emb_dim'   :  params.word_emb_dim   ,
        'enc_lstm_dim'   :  params.enc_lstm_dim   ,
        'n_enc_layers'   :  params.n_enc_layers   ,
        'dpout_model'    :  params.dpout_model    ,
        'dpout_fc'       :  params.dpout_fc       ,
        'fc_dim'         :  params.fc_dim         ,
        'bsize'          :  params.batch_size     ,
        'n_classes'      :  params.n_classes      ,
        'pool_type'      :  params.pool_type      ,
        'nonlinear_fc'   :  params.nonlinear_fc   ,
        'encoder_type'   :  params.encoder_type   ,
        'use_cuda'       :  True                  ,

    }

    # model
    encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                     'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                     'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
    assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                                 str(encoder_types)
    nli_net = NLINet(config_nli_model, weights=word_vec)

    dev_file = os.path.join('mnli_m_dev_exp','gen_mnli_rand_test.csv')
    samples_save_path = os.path.join('mnli_m_dev_exp',params.nlipath+'_m_dev_'+params.encoder_type+'_rand.jsonl')
    getScores(nli_net, dev = dev_file, dataset = params.nlipath, samples_file = samples_save_path)

    return 0

t_encoders =['InferSent'] #[ 'BLSTMprojEncoder', 'BGRUlastEncoder',
                  # 'InnerAttentionMILAEncoder','InnerAttentionNAACLEncoder',
                  # 'LSTMEncoder', 'InferSent', 'ConvNetEncoder']
t_seeds = [100]
t_dataset = ['mnli']#,'snli']#, 'personachat', 'dailydialog']
num_classes_ = {'scitail':2, 'snli':5, 'mnli': 3, 'anli': 3, 'fever': 3}
best_hyperparameters = None
best_accuracy = 0
# A list holding the object IDs for all of the experiments that we have
# launched but have not yet been processed.
remaining_ids = []
# A dictionary mapping an experiment's object ID to its hyperparameters.
# hyerparameters used for that experiment.
hyperparameters_mapping = {}

for s,d,m in itertools.product(t_seeds,t_dataset,t_encoders):
    config = {}
    config['encoder_type'] = m
    config['seed'] = s
    config['dataset'] = d
    config['num_classes'] = num_classes_[d]
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
    print("""We achieve {:7.2f}% BLEU in {} with:
        seed: {}
        model: {}
      """.format(accuracy, hyperparameters["dataset"],
                 hyperparameters["seed"], hyperparameters["encoder_type"]))
