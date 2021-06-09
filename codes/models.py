# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
## Load all kinds of model here
import torch
from tqdm.auto import tqdm

# from fairseq.models.roberta import RobertaModel


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res


def prepare_batches_fairseq(
    data,
    model,
    batch_size,
    sent1_label="sentence1",
    sent2_label="sentence2",
    target_label="label",
):
    print("preparing data ...")
    pb = tqdm(total=len(data))
    batches = []
    for start_id in range(0, len(data), batch_size):
        rows = data[start_id : start_id + batch_size]
        batch_of_pairs = [[row[sent1_label], row[sent2_label]] for row in rows]
        batch = collate_tokens(
            [model.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1,
        )
        meta = [
            [row["uid"], row[sent1_label], row[sent2_label], row[target_label]]
            for row in rows
        ]
        batches.append((batch, meta))
        pb.update(len(batch))
    pb.close()
    return batches


class NLI_Model:
    def __init__(self, args, **kwargs):
        self.args = args
        self.model = None
        self.label_fn = {}

    def predict_single(self, inp):
        pass

    def prepare_batches(self, data, batch_size=1, **kwargs):
        """prepare and return batches
        batches contain both batch of tokens and metadata

        Args:
            data ([type]): [description]
            batch_size (int, optional): [description]. Defaults to 1.
        """
        pass

    def predict_batch(self, inp):
        pass


class HubModel(NLI_Model):
    def __init__(self, args):
        super().__init__(args)
        # if args.load_model:
        print("Loading model")
        self.model = torch.hub.load(args.git_repo, args.model_name)
        self.model.eval()
        self.model.cuda()
        self.label_fn = {0: "c", 1: "n", 2: "e"}

    def prepare_batches(
        self,
        data,
        batch_size=4,
        sent1_label="premise",
        sent2_label="hypothesis",
        target_label="",
    ):
        return prepare_batches_fairseq(
            data,
            self.model,
            batch_size,
            sent1_label=sent1_label,
            sent2_label=sent2_label,
            target_label=target_label,
        )

    def predict_batch(self, batch):
        prediction_logits = self.model.predict("mnli", batch)
        prediction = prediction_logits.argmax(dim=1)
        return prediction.tolist(), prediction_logits.tolist()


class FairSeqModel(NLI_Model):
    def __init__(self, args):
        super().__init__(args)
        if "roberta" in args.model_name:
            print(f"data: {args.data_name_or_path}")
            roberta = RobertaModel.from_pretrained(
                args.model_loc,
                checkpoint_file=args.checkpoint_file,
                data_name_or_path=args.data_name_or_path,
            )
            roberta.eval()
            self.model = roberta
        else:
            raise NotImplementedError(f"{args.model_type} not yet implemented")

        self.label_fn = lambda label: self.model.task.label_dictionary.string(
            [label + self.model.task.label_dictionary.nspecial]
        )

    def prepare_batches(
        self, data, batch_size=4, sent1_label="", sent2_label="", target_label=""
    ):
        print(sent1_label, sent2_label)
        return prepare_batches_fairseq(
            data,
            self.model,
            batch_size,
            sent1_label=sent1_label,
            sent2_label=sent2_label,
            target_label=target_label,
        )

    def predict_batch(self, batch):
        prediction_logits = self.model.predict("sentence_classification_head", batch)
        prediction = prediction_logits.argmax(dim=1)
        return prediction.tolist(), prediction_logits.tolist()


class HFModel(NLI_Model):
    def __init__(self, args):
        super().__init__(args)

    def prepare_batches(self, data, batch_size, **kwargs):
        raise NotImplementedError("Not implemented in this model, run ANLI scripts...")

    def predict_batch(self, inp):
        raise NotImplementedError("Not implemented in this model, run ANLI scripts...")
