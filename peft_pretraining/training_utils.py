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
