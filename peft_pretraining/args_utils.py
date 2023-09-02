import os, sys
import yaml
from datetime import datetime

from loguru import logger


def check_args_torchrun_main(args):
    if args.training_config is not None:
        logger.info(f"Yaml config provided for the run. The file {args.training_config} is used to provide all the parameters.")
        if len(sys.argv) > 3:
            logger.error(f"argv length is {len(sys.argv)}")
            raise RuntimeError(
                "You provided both a yaml config and command line arguments. "
                "Please use only one of the two options."
            )
        with open(args.training_config) as f:
            training_config = yaml.safe_load(f)
        for k, v in training_config.items():
            if k == "lr": v = float(v)
            setattr(args, k, v)

    if (args.dataset_path is None) == (args.megatron_dataset_config is None):
        raise ValueError("Either --dataset_path or --megatron_dataset_config must be specified and not both\n"
                         f"Got {args.dataset_path=} and {args.megatron_dataset_config=}")

    if args.megatron_dataset_config is not None:
        if not os.path.exists(args.megatron_dataset_config):
            raise ValueError(f"{args.megatron_dataset_config=} does not exist")

    if args.batch_size is None:
        raise ValueError("batch_size must be specified")

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

    if args.warmed_up_model is not None:
        assert os.path.exists(args.warmed_up_model), f"{args.warmed_up_model=} does not exist"

    if args.dtype in ["fp16", "float16"]:
        raise NotImplementedError("fp16 is not supported in torchrun_main.py. Use deepspeed_main.py instead (but it seems to have bugs)")

    if (int(args.reset_optimizer_on_relora) +
        int(bool(args.optimizer_random_pruning)) +
        int(bool(args.optimizer_magnitude_pruning))
        ) > 1:
        raise ValueError("reset_optimizer_on_relora, and keep_first_opt_rows are mutually exclusive")

    if args.relora and not args.use_peft:
        logger.warning("--relora assumes --use_peft. Setting --use_peft=True")
        args.use_peft = True

    assert 0 <= args.optimizer_random_pruning < 1, "--optimizer_random_pruning must be between 0 and 1"
    assert 0 <= args.optimizer_magnitude_pruning < 1, "--optimizer_magnitude_pruning must be between 0 and 1"


    if args.distributed_type == "fsdp" and args.weight_decay > 0:
        raise ValueError("FSDP does not support weight decay yet.")

    if args.distributed_type == "fsdp" and "zero" in args.optimizer:
        raise ValueError("FSDP does zero-optimization by default, do not specify optimizer as zero optimizer.")

    if args.relora:
        if args.cycle_length is not None and args.cycle_length != args.relora:
            logger.warning(f"Overriding --cycle_length ({args.cycle_length}) to be equal to --relora ({args.relora})")
        args.cycle_length = args.relora

    return args
