# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
from nltk.translate.bleu_score import sentence_bleu as bleu_score

def isCharOnly(s):

        for c in s:
            if ord('a') <= ord(c) and ord('z') >= ord(c):
                pass
            else:
                return False

        return True

def getBLEU(fp, increment=4):

    D = fp.readlines()

    meteor_s = 0.0
    sent_bleu = 0.0
    cnt_ = 1e-3
    i=0

    while i<len(D):

        tar = D[i + 2].split()[2:]
        mod = D[i + 1].split()[1:]

        if '<eor>' in tar:
            ind_tar = tar.index('<eor>')
        else:
            ind_tar = -1

        if '<eor>' in mod:
            ind_mod = mod.index('<eor>')
        else:
            ind_mod = -1

        sent_bleu += bleu_score([mod[:ind_mod]], tar[:ind_tar], (0.5,0.5))

        i += increment
        cnt_ += 1

    return sent_bleu/float(cnt_)
