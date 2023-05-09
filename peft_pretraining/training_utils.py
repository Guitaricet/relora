import torch
import transformers


def get_scheculer(optimizer, scheduler_type, num_training_steps, warmup_steps):
    if scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    if scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented")


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
