import os
from datetime import datetime

from loguru import logger


def check_args_torchrun_main(args):
    if not args.train_ln:
        logger.error("Are you sure? Not training LN is a bad idea.")
        raise ValueError("Are you sure? Not training LN is a bad idea.")

    if args.save_dir is None:
        # use checkpoints / model name, date and time as save directory
        args.save_dir = f"checkpoints/{args.model_config.split('/')[-1].rstrip('.json')}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    if args.tags is not None:
        args.tags = args.tags.split(",")

    if not args.use_peft:
        # just for more clear hparam logging to wandb
        args.relora = None
        args.lora_r = None
        args.force_keep_original = False

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Training for {args.num_training_steps} update steps")

    if args.continue_from is not None:
        assert os.path.exists(args.continue_from), f"--continue_from={args.continue_from} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported in torchrun_main.py. Use deepspeed_main.py instead (but it seems to have bugs)")

    if (int(args.reset_optimizer_on_relora) +
        int(bool(args.svd_optimizer_on_relora)) +
        int(bool(args.keep_first_opt_rows)) +
        int(bool(args.optimizer_random_pruning)) +
        int(bool(args.optimizer_magnitude_pruning))
        ) > 1:
        raise ValueError("reset_optimizer_on_relora, svd_optimizer_on_relora, and keep_first_opt_rows are mutually exclusive")

    if args.relora and not args.use_peft:
        logger.warning("--relora assumes --use_peft. Setting --use_peft=True")
        args.use_peft = True

    assert 0 <= args.optimizer_random_pruning < 1, "--optimizer_random_pruning must be between 0 and 1"
    assert 0 <= args.optimizer_magnitude_pruning < 1, "--optimizer_magnitude_pruning must be between 0 and 1"

    return args
