# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under Creative Commons-Non Commercial 4.0 found in the
# LICENSE file in the root directory of this source tree.
#
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
"""

import torch
import logging
import math
from pathlib import Path
import os
from dataclasses import dataclass, field
from typing import Optional
from src.nli.training import MODEL_CLASSES
import src.config as nli_config
import copy
import submitit
from tqdm.auto import tqdm
import numpy as np
import json

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    ## NLI model params
    load_nli_model: bool = field(default=False, metadata={"help": "Load NLI model"})

    model_class_name: Optional[str] = field(default="roberta-base")

    model_checkpoint_path: Optional[str] = field(default="roberta-base")

    slurm: bool = field(default=False, metadata={"help": "Run job in submitit"})

    output_file: Optional[str] = field(default="eval_results_lm.json")

    checkpoint_file: Optional[str] = field(default="")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )

    mlm: bool = field(
        default=False,
        metadata={
            "help": "Train with masked-language modeling loss instead of language modeling."
        },
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5,
        metadata={
            "help": "Maximum length of a span of masked tokens for permutation language modeling."
        },
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(
            tokenizer=tokenizer, file_path=file_path, block_size=args.block_size
        )
    else:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=args.block_size,
            overwrite_cache=args.overwrite_cache,
            cache_dir=cache_dir,
        )


def main(model_args, data_args, training_args):

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    # modify model loading here
    # if model_args.model_name_or_path:
    logger.info(f"Initializing Causal LM model ...")
    # config.is_decoder = True
    model = AutoModelForCausalLM.from_config(config)
    if model_args.load_nli_model:
        logger.info(f"Loading NLI model {model_args.model_class_name}")
        model_class_item = MODEL_CLASSES[model_args.model_class_name]
        model_name = model_class_item["model_name"]
        nli_model = model_class_item["sequence_classification"].from_pretrained(
            model_name, cache_dir=str(nli_config.PRO_ROOT / "trans_cache"), num_labels=3
        )
        logger.info(
            f"Loading NLI pre-trained weights from {model_args.model_checkpoint_path}"
        )
        nli_model.load_state_dict(torch.load(model_args.model_checkpoint_path))
        model.base_model.load_state_dict(nli_model.base_model.state_dict())
    else:
        logger.info("Not loading any NLI models, using the previous pre-trained models")
    # Freeze encoder params
    logger.info("Freezing params ...")
    for param in model.base_model.parameters():
        param.requires_grad = False
    tot_param = [n for n, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Total learnable params = {len(tot_param)}")
    logger.info(",".join(tot_param))
    # else:
    #     logger.info("Training new model from scratch")
    #     model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # if (
    #     config.model_type in ["bert", "roberta", "distilbert", "camembert"]
    #     and not data_args.mlm
    # ):
    #     raise ValueError(
    #         "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
    #         "--mlm flag (masked language modeling)."
    #     )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
        if training_args.do_train
        else None
    )
    if train_dataset is not None:
        logger.info(f"Read training dataset : {len(train_dataset)}")
    eval_dataset = (
        get_dataset(
            data_args,
            tokenizer=tokenizer,
            evaluate=True,
            cache_dir=model_args.cache_dir,
        )
        if training_args.do_eval
        else None
    )
    if eval_dataset is not None:
        logger.info(f"Read eval dataset : {len(eval_dataset)}")
    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=data_args.mlm,
            mlm_probability=data_args.mlm_probability,
        )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None
            and os.path.isdir(model_args.model_name_or_path)
            else None
        )

        # def my_hp_space(trial):
        #     return {
        #         "learning_rate": trial.suggest_float(
        #             "learning_rate", 1e-4, 1e-2, log=True
        #         ),
        #         "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        #         "per_device_train_batch_size": trial.suggest_categorical(
        #             "per_device_train_batch_size", [4, 8, 16, 32, 64]
        #         ),
        #     }

        # trainer.hyperparameter_search(direction="maximize", hp_space=my_hp_space)

        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        ## load the saved decoder
        logger.info("Loading decoder...")
        if len(model_args.checkpoint_file) == 0:
            model.load_state_dict(
                torch.load(Path(training_args.output_dir) / "pytorch_model.bin")
            )
        else:
            model.load_state_dict(
                torch.load(
                    Path(training_args.output_dir)
                    / model_args.checkpoint_file
                    / "pytorch_model.bin"
                )
            )
        logger.info("*** Evaluate ***")
        print(f"Eval batch size : {trainer.args.eval_batch_size}")
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
        prediction_loss_only = True
        eval_perplexity = []
        for inputs in tqdm(eval_dataloader, desc="Eval"):
            loss, logits, labels = trainer.prediction_step(
                model, inputs, prediction_loss_only
            )
            eval_perplexity.append(math.exp(loss))

        # eval_output = trainer.evaluate()

        # perplexity = math.exp(eval_output["eval_loss"])
        result = {
            "perplexity": eval_perplexity,
            "mean_perplexity": np.mean(eval_perplexity),
        }

        output_eval_file = model_args.output_file
        if trainer.is_world_master():
            json.dump(result, open(output_eval_file, "w"))
            logger.info("***** Eval results *****")
            logger.info(f"Perplexity : {result['mean_perplexity']}")

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.slurm:
        # run by submitit
        submitit_logdir = Path(training_args.logging_dir) / "submitit_logs"
        executor = submitit.AutoExecutor(folder=submitit_logdir)
        executor.update_parameters(
            timeout_min=720,
            slurm_partition="learnfair",
            gpus_per_node=8,
            tasks_per_node=1,
            cpus_per_task=10,
            slurm_mem="",
        )
        job = executor.submit(main, model_args, data_args, training_args)
        print(f"Submitted job {job.job_id}")
    else:
        main(model_args, data_args, training_args)
