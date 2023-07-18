import itertools
import multiprocessing
from itertools import chain

import torch
from torch.utils.data import IterableDataset, get_worker_info
from transformers import AutoTokenizer, default_data_collator
from datasets import Dataset


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for example in iter_data:
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
        attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def tokenize_and_chunk(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    text_field: str,
    sequence_length: int,
    num_cpu: int = multiprocessing.cpu_count(),
):
    """
    Build data loaders for training.

    This function performs the following steps:
    1. Load the tokenizer from the pretrained "EleutherAI/gpt-neox-20b" model.
    2. Load the "openwebtext" dataset.
    3. Tokenize the dataset, adding the end-of-sentence token to each text.
    4. Process the tokenized dataset into chunks of a specified block size.

    Returns:
        Dataset: The processed dataset ready for training.
    """
    extra_map_kwargs = {"num_proc": num_cpu}  # iterable dataset does not support workers in map
    if isinstance(dataset, IterableDataset):
        extra_map_kwargs = {}

    tokenized_dataset = dataset.map(
        lambda example: tokenizer([t + tokenizer.eos_token for t in example[text_field]]),
        batched=True,
        remove_columns=[text_field],
        **extra_map_kwargs,
    )

    block_size = sequence_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    train_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        **extra_map_kwargs,
    )

    return train_dataset
