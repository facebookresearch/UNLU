# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
import re, math, time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nopeak_mask(size):
    "Mask out subsequent positions. aka subsequent_mask"
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  torch.from_numpy(np_mask) == 0
    if device == torch.device("cuda:0") and next(model.parameters()).is_cuda:
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, pad_idx):

    src_mask = (src != pad_idx).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size).to(device)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask
