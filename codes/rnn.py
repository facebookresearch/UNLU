# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    ''' This class contains the implementation of Decoder Module.
    Args:
        embedding_dim: A integer indicating the embedding size.
        output_dim: A integer indicating the size of output dimension.
        hidden_dim: A integer indicating the hidden size of rnn.
        n_layers: A integer indicating the number of layers in rnn.
        dropout: A float indicating the dropout.
    '''
    def __init__(self, embedding_dim, output_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first = False).to(device)
        self.linear = nn.Linear(hidden_dim, output_dim).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, input, hidden, cell):
        # input is of shape [batch_size]
        # hidden is of shape [n_layer * num_directions, batch_size, hidden_size]
        # cell is of shape [n_layer * num_directions, batch_size, hidden_size]

        input = input.unsqueeze(0)
        # input shape is [1, batch_size]. reshape is needed rnn expects a rank 3 tensors as input.
        # so reshaping to [1, batch_size] means a batch of batch_size each containing 1 index.

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # embedded is of shape [1, batch_size, embedding_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # generally output shape is [sequence_len, batch_size, hidden_dim * num_directions]
        # generally hidden shape is [num_layers * num_directions, batch_size, hidden_dim]
        # generally cell shape is [num_layers * num_directions, batch_size, hidden_dim]

        # sequence_len and num_directions will always be 1 in the decoder.
        # output shape is [1, batch_size, hidden_dim]
        # hidden shape is [num_layers, batch_size, hidden_dim]
        # cell shape is [num_layers, batch_size, hidden_dim]
        predicted = F.log_softmax(self.linear(output), dim = 2) # linear expects as rank 2 tensor as input
        # predicted shape is [batch_size, output_dim]

        return predicted, hidden, cell


class AttnDecoder(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_dim, n_layers, dropout, max_length):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.num_layers = n_layers
        self.max_length = max_length
        self.dropout_p = dropout
        self.attn = nn.Linear(self.hidden_size + embedding_dim, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size + embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, self.num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input)
        encoder_outputs = encoder_outputs.view(-1, self.hidden_size, self.max_length)
        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1).unsqueeze(0).view(-1, self.max_length, 1)
        #encoder_outputs = encoder_outputs.view(-1, self.hidden_size, self.max_length)
        attn_applied = torch.bmm(encoder_outputs, attn_weights)
        output = torch.cat((embedded, attn_applied[:, :, 0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))
        predicted = F.log_softmax(self.linear(output), dim = 2)
        return predicted, hidden, cell

class RecurrentEncoder(nn.Module):
    ''' Sequence to sequence networks consists of Encoder and Decoder modules.
    This class contains the implementation of Encoder module.
    Args:
        input_dim: A integer indicating the size of input dimension.
        emb_dim: A integer indicating the size of embeddings.
        hidden_dim: A integer indicating the hidden dimension of RNN layers.
        n_layers: A integer indicating the number of layers.
        dropout: A float indicating dropout.
    '''
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, bi_directional=False):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=False)
        self.hrnn = nn.LSTM(hidden_dim,hidden_dim, n_layers, dropout = dropout, bidirectional = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src is of shape [sentence_length, batch_size], it is time major

        # embedded is of shape [sentence_length, batch_size, embedding_size]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # Decode the hidden state of the last time step
        # inputs to the rnn is input, (h, c); if hidden, cell states are not passed means default initializes to zero.
        # input is of shape [sequence_length, batch_size, input_size]
        # hidden is of shape [num_layers * num_directions, batch_size, hidden_size]
        # cell is of shape [num_layers * num_directions, batch_size, hidden_size]
        outputs, (hidden, cell) = self.rnn(embedded)
        outputs, (hidden, cell) = self.hrnn(outputs)
        # outputs are always from the top hidden layer, if bidirectional outputs are concatenated.
        # outputs shape [sequence_length, batch_size, hidden_dim * num_directions]
        return outputs, hidden, cell


class Encoder(nn.Module):
    ''' Sequence to sequence networks consists of Encoder and Decoder modules.
    This class contains the implementation of Encoder module.
    Args:
        input_dim: A integer indicating the size of input dimension.
        emb_dim: A integer indicating the size of embeddings.
        hidden_dim: A integer indicating the hidden dimension of RNN layers.
        n_layers: A integer indicating the number of layers.
        dropout: A float indicating dropout.
    '''
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, bi_directional=False):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bi_directional = bi_directional
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=bi_directional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src is of shape [sentence_length, batch_size], it is time major

        # embedded is of shape [sentence_length, batch_size, embedding_size]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        # Decode the hidden state of the last time step
        # inputs to the rnn is input, (h, c); if hidden, cell states are not passed means default initializes to zero.
        # input is of shape [sequence_length, batch_size, input_size]
        # hidden is of shape [num_layers * num_directions, batch_size, hidden_size]
        # cell is of shape [num_layers * num_directions, batch_size, hidden_size]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs are always from the top hidden layer, if bidirectional outputs are concatenated.
        # outputs shape [sequence_length, batch_size, hidden_dim * num_directions]
        if self.bi_directional:
            outputs = outputs[:, :, self.hidden_dim:] + outputs[:, :, :self.hidden_dim]
            hidden = hidden[:2,:,:] + hidden[2:,:,:]
            cell = cell[:2,:,:] + cell[2:,:,:]
            #hidden = hidden.view(self.n_layers,-1,self.hidden_dim)
            #cell = cell.view(self.n_layers,-1,self.hidden_dim)
        return outputs, hidden, cell
