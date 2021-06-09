# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This file contains the definition of encoders used in https://arxiv.org/pdf/1705.02364.pdf
"""

import numpy as np
import time

import torch
import torch.nn as nn

device = 'cpu'
#if torch.cuda.is_available():
#    device = 'cuda'

"""
BLSTM (max/mean) encoder
"""

class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, config['n_enc_layers'],
                                bidirectional=True, dropout=self.dpout_model)

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = torch.sort(sent_len, descending=True)
        #sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = torch.sort(idx_sort)[1]

        idx_sort = idx_sort.to(device)
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted).to(device)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = idx_unsort.to(device)
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            #sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).to(device)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb, sent_output.permute(1,0,2)

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = torch.sort(lengths)[0], torch.sort(-lengths)[1]
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx:stidx + bsize])
            if self.is_cuda():
                batch = batch.to(device)
            with torch.no_grad():
                batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = torch.sort(idx_sort)[1]
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

    def visualize(self, sent, tokenize=True):

        sent = sent.split() if not tokenize else self.tokenize(sent)
        sent = [[self.bos] + [word for word in sent if word in self.word_vec] + [self.eos]]

        if ' '.join(sent[0]) == '%s %s' % (self.bos, self.eos):
            import warnings
            warnings.warn('No words in "%s" have w2v vectors. Replacing \
                           by "%s %s"..' % (sent, self.bos, self.eos))
        batch = self.get_batch(sent)

        if self.is_cuda():
            batch = batch.to(device)
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        # output, idxs = output.squeeze(), idxs.squeeze()
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

        # visualize model
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0 * n / np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()

        return output, idxs

"""
BiGRU encoder (first/last hidden states)
"""


class BGRUlastEncoder(nn.Module):
    def __init__(self, config):
        super(BGRUlastEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.GRU(self.word_emb_dim, self.enc_lstm_dim, config['n_enc_layers'],
                               bidirectional=True, dropout=self.dpout_model)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: seqlen x batch x worddim

        sent, sent_len = sent_tuple
        sent_len = sent_len.cpu()
        # Sort by length (keep idx)
        #sent_len, idx_sort = torch.sort(sent_len, descending=True)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.LongTensor(idx_sort).to(device))
        sent_len = np.array(sent_len)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output, hn = self.enc_lstm(sent_packed)
        emb = torch.cat((hn[0], hn[1]), 1)  # batch x 2*nhid
        sent_output_un, _ = torch.nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)

        # Un-sort by length
        idx_unsort = torch.sort(idx_sort)[1]
        emb = emb.index_select(0, torch.LongTensor(idx_unsort).to(device))

        return emb, sent_output_un


"""
BLSTM encoder with projection after BiLSTM
"""


class BLSTMprojEncoder(nn.Module):
    def __init__(self, config):
        super(BLSTMprojEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, config['n_enc_layers'],
                                bidirectional=True, dropout=self.dpout_model)
        self.proj_enc = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, bias=False)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: (seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)
        sent_len = sent_len.cpu()
        # Sort by length (keep idx)
        #sent_len, idx_sort = torch.sort(sent_len, descending=True)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.LongTensor(idx_sort).to(device))
        sent_len = np.array(sent_len)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed)[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1, torch.LongTensor(idx_unsort).to(device))

        sent_output = self.proj_enc(sent_output.view(-1, 2*self.enc_lstm_dim)).view(-1, bsize, 2*self.enc_lstm_dim)
        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len).unsqueeze(1).to(device)
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            emb = torch.max(sent_output, 0)[0].squeeze(0)

        return emb, sent_output.permute(1,0,2)


"""
LSTM encoder
"""


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, config['n_enc_layers'],
                                bidirectional=False, dropout=self.dpout_model)

    def forward(self, sent_tuple):
        # sent_len [max_len, ..., min_len] (batch)
        # sent (seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        sent_len = sent_len.cpu()

        # Sort by length (keep idx)
        #sent_len, idx_sort = torch.sort(sent_len, descending=True)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.LongTensor(idx_sort).to(device))

        # Handling padding in Recurrent Networks
        sent_len = np.array(sent_len)
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output, hn = self.enc_lstm(sent_packed)  # batch x 2*nhid
        sent_hn = hn[0].squeeze(0)
        # Un-sort by length
        sent_output_un, _ = torch.nn.utils.rnn.pad_packed_sequence(sent_output, batch_first=True)
        idx_unsort = np.argsort(idx_sort)
        emb = sent_hn.index_select(0, torch.LongTensor(idx_unsort).to(device))

        return emb, sent_output_un


"""
GRU encoder
"""


class GRUEncoder(nn.Module):
    def __init__(self, config):
        super(GRUEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim =  config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.GRU(self.word_emb_dim, self.enc_lstm_dim, config['n_enc_layers'],
                               bidirectional=False, dropout=self.dpout_model)

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: (seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        sent_len = sent_len.cpu()

        # Sort by length (keep idx)
        #sent_len, idx_sort = torch.sort(sent_len, descending=True)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.LongTensor(idx_sort).to(device))
        sent_len = np.array(sent_len)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)

        sent_output, hn = self.enc_lstm(sent_packed)
        sent_hn = hn.squeeze(0)

        # batch x 2*nhid

        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        emb = sent_hn.index_select(0, torch.LongTensor(idx_unsort).to(device))

        return emb, sent_output


"""
Inner attention from "hierarchical attention for document classification"
"""


class InnerAttentionNAACLEncoder(nn.Module):
    def __init__(self, config):
        super(InnerAttentionNAACLEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']


        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, config['n_enc_layers'], bidirectional=True)

        self.proj_key = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, bias=False)
        self.proj_lstm = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, bias=False)
        self.query_embedding = nn.Embedding(1, 2*self.enc_lstm_dim)
        self.softmax = nn.Softmax()

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: (seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)
        sent_len = sent_len.cpu()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.LongTensor(idx_sort).to(device))
        sent_len = np.array(sent_len)
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed)[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1, torch.LongTensor(idx_unsort).to(device))

        sent_output = sent_output.transpose(0,1).contiguous()

        sent_output_proj = self.proj_lstm(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)

        sent_key_proj = self.proj_key(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)

        sent_key_proj = torch.tanh(sent_key_proj)
        # NAACL paper: u_it=tanh(W_w.h_it + b_w)  (bsize, seqlen, 2nhid)

        sent_w = self.query_embedding(torch.LongTensor(bsize*[0]).to(device)).unsqueeze(2) #(bsize, 2*nhid, 1)

        Temp = 2
        keys = sent_key_proj.bmm(sent_w).squeeze(2) / Temp

        # Set probas of padding to zero in softmax
        keys = keys + ((keys == 0).float()*-10000)

        alphas = self.softmax(keys/Temp).unsqueeze(2).expand_as(sent_output)
        # if int(time.time()) % 100 == 0:
        #     print('w', torch.max(sent_w), torch.min(sent_w))
        #     print('alphas', alphas[0, :, 0])
        emb = torch.sum(alphas * sent_output_proj, 1).squeeze(1)

        return emb, sent_output


"""
Inner attention inspired from "Self-attentive ..."
"""


class InnerAttentionMILAEncoder(nn.Module):
    def __init__(self, config):
        super(InnerAttentionMILAEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim =  config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, config['n_enc_layers'], bidirectional=True)

        self.proj_key = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, bias=False)
        self.proj_lstm = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, bias=False)
        self.query_embedding = nn.Embedding(2, 2*self.enc_lstm_dim)
        self.softmax = nn.Softmax()

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: (seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)
        sent_len = sent_len.cpu()
        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.LongTensor(idx_sort).to(device))
        sent_len = np.array(sent_len)
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed)[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1, torch.LongTensor(idx_unsort).to(device))

        sent_output = sent_output.transpose(0,1).contiguous()
        sent_output_proj = self.proj_lstm(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)
        sent_key_proj = self.proj_key(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)
        sent_key_proj = torch.tanh(sent_key_proj)
        # NAACL : u_it=tanh(W_w.h_it + b_w) like in NAACL paper

        # Temperature
        Temp = 3

        sent_w1 = self.query_embedding(torch.LongTensor(bsize*[0]).to(device)).unsqueeze(2) #(bsize, nhid, 1)
        keys1 = sent_key_proj.bmm(sent_w1).squeeze(2) / Temp
        keys1 = keys1 + ((keys1 == 0).float()*-1000)
        alphas1 = self.softmax(keys1).unsqueeze(2).expand_as(sent_key_proj)
        emb1 = torch.sum(alphas1 * sent_output_proj, 1).squeeze(1)


        sent_w2 = self.query_embedding(torch.LongTensor(bsize*[1]).to(device)).unsqueeze(2) #(bsize, nhid, 1)
        keys2 = sent_key_proj.bmm(sent_w2).squeeze(2) / Temp
        keys2 = keys2 + ((keys2 == 0).float()*-1000)
        alphas2 = self.softmax(keys2).unsqueeze(2).expand_as(sent_key_proj)
        emb2 = torch.sum(alphas2 * sent_output_proj, 1).squeeze(1)

        sent_w3 = self.query_embedding(torch.LongTensor(bsize*[1]).to(device)).unsqueeze(2) #(bsize, nhid, 1)
        keys3 = sent_key_proj.bmm(sent_w3).squeeze(2) / Temp
        keys3 = keys3 + ((keys3 == 0).float()*-1000)
        alphas3 = self.softmax(keys3).unsqueeze(2).expand_as(sent_key_proj)
        emb3 = torch.sum(alphas3 * sent_output_proj, 1).squeeze(1)

        sent_w4 = self.query_embedding(torch.LongTensor(bsize*[1]).to(device)).unsqueeze(2) #(bsize, nhid, 1)
        keys4 = sent_key_proj.bmm(sent_w4).squeeze(2) / Temp
        keys4 = keys4 + ((keys4 == 0).float()*-1000)
        alphas4 = self.softmax(keys4).unsqueeze(2).expand_as(sent_key_proj)
        emb4 = torch.sum(alphas4 * sent_output_proj, 1).squeeze(1)


        #if int(time.time()) % 100 == 0:
        #    print('alphas', torch.cat((alphas1.data[0, :, 0],
        #                               alphas2.data[0, :, 0],
        #                               torch.abs(alphas1.data[0, :, 0] -
        #                                         alphas2.data[0, :, 0])), 1))

        emb = torch.cat((emb1, emb2, emb3, emb4), 1)
        return emb, sent_output


"""
Inner attention from Yang et al.
"""


class InnerAttentionYANGEncoder(nn.Module):
    def __init__(self, config):
        super(InnerAttentionYANGEncoder, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, config['n_enc_layers'],
                                bidirectional=True)

        self.proj_lstm = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                   bias=True)
        self.proj_query = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                    bias=True)
        self.proj_enc = nn.Linear(2*self.enc_lstm_dim, 2*self.enc_lstm_dim,
                                  bias=True)

        self.query_embedding = nn.Embedding(1, 2*self.enc_lstm_dim)
        self.softmax = nn.Softmax()

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: (seqlen x batch x worddim)

        sent, sent_len = sent_tuple
        bsize = sent.size(1)
        sent_len = sent_len.cpu()

        # Sort by length (keep idx)
        sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent = sent.index_select(1, torch.LongTensor(idx_sort).to(device))
        sent_len = np.array(sent_len)
        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len)
        sent_output = self.enc_lstm(sent_packed)[0]
        # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]
        # Un-sort by length
        idx_unsort = np.argsort(idx_sort)
        sent_output = sent_output.index_select(1, torch.LongTensor(idx_unsort).to(device))

        sent_output = sent_output.transpose(0,1).contiguous()

        sent_output_proj = self.proj_lstm(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)

        sent_keys = self.proj_enc(sent_output.view(-1,
            2*self.enc_lstm_dim)).view(bsize, -1, 2*self.enc_lstm_dim)

        sent_max = torch.max(sent_output, 1)[0].squeeze(1)  # (bsize, 2*nhid)
        sent_summary = self.proj_query(sent_max).unsqueeze(1).expand_as(sent_keys)
        # (bsize, seqlen, 2*nhid)

        sent_M = torch.tanh(sent_keys + sent_summary)
        # (bsize, seqlen, 2*nhid) YANG : M = tanh(Wh_i + Wh_avg
        sent_w = self.query_embedding(torch.LongTensor(bsize*[0]).to(device)).unsqueeze(2)
        # (bsize, 2*nhid, 1)

        sent_alphas = self.softmax(sent_M.bmm(sent_w).squeeze(2)).unsqueeze(1)
        # (bsize, 1, seqlen)

        # if int(time.time()) % 200 == 0:
        #     print('w', torch.max(sent_w[0]), torch.min(sent_w[0]))
        #     print('alphas', sent_alphas[0][0][0:sent_len[0]])
        # # Get attention vector
        emb = sent_alphas.bmm(sent_output_proj).squeeze(1)

        return emb, sent_output



"""
Hierarchical ConvNet
"""
class ConvNetEncoder(nn.Module):
    def __init__(self, config):
        super(ConvNetEncoder, self).__init__()

        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']

        self.convnet1 = nn.Sequential(
            nn.Conv1d(self.word_emb_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.convnet2 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.convnet3 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )
        self.convnet4 = nn.Sequential(
            nn.Conv1d(2*self.enc_lstm_dim, 2*self.enc_lstm_dim, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            )



    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (batch)
        # sent: (seqlen x batch x worddim)

        sent, sent_len = sent_tuple

        sent = sent.transpose(0,1).transpose(1,2).contiguous()
        # batch, nhid, seqlen)

        sent = self.convnet1(sent)
        u1 = torch.max(sent, 2)[0]

        sent = self.convnet2(sent)
        u2 = torch.max(sent, 2)[0]

        sent = self.convnet3(sent)
        u3 = torch.max(sent, 2)[0]

        sent = self.convnet4(sent)
        u4 = torch.max(sent, 2)[0]

        emb = torch.cat((u1, u2, u3, u4), 1)

        return emb, sent.permute(0,2,1)


"""
BiLSTM
"""

class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM,self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 2,
                                bidirectional=True, dropout=self.dpout_model)
        self.relu = nn.ReLU()
        self.projection = nn.Linear(self.word_emb_dim, self.enc_lstm_dim)

    def forward(self,sent_tuple):
        sent,sent_len = sent_tuple
        sent_len = sent_len.cpu()
        bsize = sent.size(1)

        sent_proj = self.relu(self.projection(sent))

        out, (emb_ht,_) = self.enc_lstm(sent_proj)
        emb = emb_ht[-2:].transpose(0, 1).contiguous().view(bsize, -1)
        return emb,out


"""
Main module for Natural Language Inference
"""


class NLINet(nn.Module):
    def __init__(self, config, weights = None):
        super(NLINet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.embedding = nn.Embedding(config['n_words'], config['word_emb_dim']).to(device)
        if weights is not None:
            self.embedding.load_state_dict({'weight':weights})
        self.embedding.weight.requires_grad = False

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 4*2*self.enc_lstm_dim
        self.inputdim = 4*self.inputdim if self.encoder_type in \
                        ["ConvNetEncoder", "InnerAttentionMILAEncoder"] else self.inputdim
        self.inputdim = self.inputdim/2 if self.encoder_type == "LSTMEncoder" \
                                        else self.inputdim
        self.inputdim = int(self.inputdim)
        
        self.lin1 = nn.Linear(self.inputdim, self.fc_dim)
        self.lin2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.lin3 = nn.Linear(self.fc_dim, self.n_classes)
        
        for lin in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

        if self.nonlinear_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.inputdim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.fc_dim),
                nn.Tanh(),
                nn.Dropout(p=self.dpout_fc),
                nn.Linear(self.fc_dim, self.n_classes),
                )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=self.dpout_fc),
                self.lin1,
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                self.lin2,
                nn.ReLU(),
                nn.Dropout(p=self.dpout_fc),
                self.lin3
                )

    def forward(self, s1, s2):
        # s1 : (s1, s1_len)
        s1_embed = self.embedding(s1[0]).to(device)
        s2_embed = self.embedding(s2[0]).to(device)
        u, s1_out = self.encoder((s1_embed,s1[1]))
        v, s2_out = self.encoder((s2_embed,s2[1]))

        features = torch.cat((u, v, torch.abs(u-v), u*v), 1)
        output = self.classifier(features)
        return output, (s1_out, s2_out)

    def encode(self, s1, is_probe = False):
        # s1 : (s1, s1_len)
        s1_embed = self.embedding(s1[0])
        emb, out = self.encoder((s1_embed,s1[1]))

        return emb, out


"""
Main module for Classification
"""


class ClassificationNet(nn.Module):
    def __init__(self, config):
        super(ClassificationNet, self).__init__()

        # classifier
        self.nonlinear_fc = config['nonlinear_fc']
        self.fc_dim = config['fc_dim']
        self.n_classes = config['n_classes']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.encoder_type = config['encoder_type']
        self.dpout_fc = config['dpout_fc']

        self.encoder = eval(self.encoder_type)(config)
        self.inputdim = 2*self.enc_lstm_dim
        self.inputdim = 4*self.inputdim if self.encoder_type == "ConvNetEncoder" else self.inputdim
        self.inputdim = self.enc_lstm_dim if self.encoder_type =="LSTMEncoder" else self.inputdim
        self.classifier = nn.Sequential(
            nn.Linear(self.inputdim, 512),
            nn.Linear(512, self.n_classes),
        )

    def forward(self, s1):
        # s1 : (s1, s1_len)
        u = self.encoder(s1)

        output = self.classifier(u)
        return output

    def encode(self, s1):
        emb, output = self.encoder(s1)
        return emb, output
