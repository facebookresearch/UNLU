# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse

import numpy as np
from comet_ml import OfflineExperiment
import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab, DICO_LABEL
from mutils import get_optimizer
from models import NLINet
import pandas as pd
from collections import Counter
import pdb


parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='datasets/snli/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--word_emb_path", type=str, default="dataset/GloVe/glove.840B.300d.txt", help="word embedding file path")

# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSentV1', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=2048, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=7, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=3, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--word_emb_type", type=str, default='normal', help="word embedding type, either glove or normal")

# comet
parser.add_argument("--comet_apikey", type=str, default='', help="comet api key")
parser.add_argument("--comet_workspace", type=str, default='', help="comet workspace")
parser.add_argument("--comet_project", type=str, default='', help="comet project name")
parser.add_argument("--comet_disabled", action='store_true', help="if true, disable comet")

params, _ = parser.parse_known_args()

# set gpu device
# torch.cuda.set_device(params.gpu_id)
exp_folder = os.path.join(params.outputdir, 'exp_seed_{}'.format(params.seed))
if not os.path.exists(exp_folder):
    os.mkdir(exp_folder)

# set proper name
last_path = params.nlipath.split('/')[-2]
save_folder_name = os.path.join(exp_folder, last_path)
if not os.path.exists(save_folder_name):
    os.mkdir(save_folder_name)
params.outputmodelname = os.path.join(save_folder_name, '{}_model.pkl'.format(params.encoder_type))
# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)
pr = vars(params)

ex = OfflineExperiment(
                workspace=pr['comet_workspace'],
                project_name=pr['comet_project'],
                disabled=pr['comet_disabled'],
                offline_directory= os.path.join(save_folder_name,'comet_runs'))

ex.log_parameters(pr)
ex.set_name(pr['encoder_type'])

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
# torch.cuda.manual_seed(params.seed)

"""
DATA
"""
train, valid, test = get_nli(params.nlipath)
word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path, emb_dim=params.word_emb_dim,
                       wtype=params.word_emb_type)

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])

# Train label class balancing
weights = [2, 2, 2, 0.3, 7, 2, 6]
# invert the weights by values



"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
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
nli_net = NLINet(config_nli_model)
print(nli_net)

# loss
weight = torch.FloatTensor(weights)
loss_fn = nn.CrossEntropyLoss(weight=weight)
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
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.to(device)), Variable(s2_batch.to(device))
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).to(device)
        k = s1_batch.size(1)  # actual batch size

        # model forward
        output, (s1_out, s2_out) = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss
        # pdb.set_trace()
        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

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
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr

        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.*correct/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * correct/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    ex.log_metric('train_accuracy', train_acc, step=epoch)
    return train_acc

inv_label = {v:k for k,v in DICO_LABEL.items()}

def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop
    test_prediction = []

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.to(device)), Variable(s2_batch.to(device))
        tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).to(device)

        # model forward
        output = nli_net((s1_batch, s1_len), (s2_batch, s2_len))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum().item()

        if eval_type == 'test':
            test_prediction.extend([inv_label[p] for p in pred.long().to('cpu').numpy()])

    # save model
    eval_acc = round(100 * correct / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
        ex.log_metric('{}_accuracy'.format(eval_type), eval_acc, step=epoch)
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))
        ex.log_metric('{}_accuracy'.format(eval_type), eval_acc, step=epoch)

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            #if not os.path.exists(params.outputdir):
            #    os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), params.outputmodelname)
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                # adam_stop = True

    if eval_type == 'test':
        target = [inv_label[t] for t in target]
        s1 = [' '.join(s) for s in s1]
        s2 = [' '.join(s) for s in s2]
        outp = pd.DataFrame({'s_1': s1, 's_2': s2, 'true_target':target, 'predicted': test_prediction})
        res_file = '{}_{}_outp.csv'.format(last_path, params.encoder_type)
        outp.to_csv(os.path.join(save_folder_name, res_file))
        #ex.log_asset(file_name=res_file, file_like_object=open(res_file,'r'))

    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.
nli_net.load_state_dict(torch.load(params.outputmodelname))

print('\nTEST : Epoch {0}'.format(epoch))
valid_acc = evaluate(1e6, 'valid', True)
test_acc = evaluate(0, 'test', True)

res_file = '{}_{}_result.txt'.format(last_path, params.encoder_type)
with open(os.path.join(save_folder_name, res_file),'w') as fp:
    fp.write("val - {}\ntest - {}\n".format(valid_acc, test_acc))
# ex.log_asset(file_name=res_file, file_like_object=open(res_file, 'r'))

# Save encoder instead of full model
torch.save(nli_net.encoder.state_dict(), params.outputmodelname)
# Save word vectors
torch.save(word_vec, params.outputmodelname + '.wordvec')
