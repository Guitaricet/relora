"""
Download and pre-tokenize a huggingface dataset.
Based on: https://github.com/conceptofmind/PaLM/blob/main/palm/build_dataloaders.py

Usage:
    python build_dataloaders.py --tokenizer EleutherAI/gpt-neox-20b --dataset openwebtext --text_field text --sequence_length 2048
"""
import os
import time
import argparse
import multiprocessing
from itertools import chain

from loguru import logger
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


from peft_pretraining.dataloader import tokenize_and_chunk


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name")
    parser.add_argument("--text_field", type=str, default="text", help="Name of the text field in the dataset")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_cpu", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples in the dataset")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the pre-tokenized dataset")

    args = parser.parse_args(args)
    return args


def main(args):
    logger.info("*" * 40)
    logger.info(f"Starting script with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    _tokenizer_name_for_save = args.tokenizer.replace("/", "_")
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{_tokenizer_name_for_save}_{args.sequence_length}")
    if os.path.exists(save_path):
        raise ValueError(f"Path {save_path} already exists")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    dataset = load_dataset(dataset, split="train")

    if args.limit is not None:
        logger.info(f"Limiting the dataset to {args.limit} examples")
        dataset = dataset.select(range(args.limit))

    logger.info("Tokenizing and chunking the dataset")
    _time = time.time()
    dataset = tokenize_and_chunk(
        tokenizer=tokenizer,
        dataset=dataset,
        text_field=args.text_field,
        sequence_length=args.sequence_length,
        num_cpu=args.num_cpu,
    )
    _hours = (time.time() - _time) / 3600
    logger.info(f"Tokenization and chunking took {_hours:.2f} hours")

    dataset.save_to_disk(save_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
