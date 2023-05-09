import math
from functools import partial

import torch
from torch.optim.lr_scheduler import LambdaLR
import transformers


def get_scheculer(optimizer, scheduler_type, num_training_steps, warmup_steps, min_lr_ratio):
    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_lr_ratio,
        )
    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


def get_cosine_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, min_lr_ratio=0.1, last_epoch=-1):
    lr_lambda = partial(
        _get_cosine_schedule_with_min_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_min_lr_lambda(current_step, *, num_warmup_steps, num_training_steps, num_cycles, min_lr_ratio):
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay


def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([example["input_ids"] for example in batch_list]),
        "attention_mask": torch.stack([example["attention_mask"] for example in batch_list]),
    }
    return batch


def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def max_train_tokens_to_number(max_train_tokens):
    if max_train_tokens.endswith("M"):
        return int(max_train_tokens.rstrip("M")) * 1_000_000
    elif max_train_tokens.endswith("B"):
        return int(max_train_tokens.rstrip("B")) * 1_000_000_000
    else:
        return int(max_train_tokens)
