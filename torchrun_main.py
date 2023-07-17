"""
Distributed training code for ReLoRA.

IMPORTANT:
The number of training steps is assumed to be smaller than the number of batches in the dataset (<= 1 epoch).
Meaning if provided with 1000000000 steps, it may stop earlier than that if the script run out of data.
"""
import os
import time
import json
import random
import hashlib
import argparse
from typing import Union
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM, LlamaDecoderLayer
from peft_pretraining.relora import ReLoRaModel, ReLoRaLinear, merge_and_reinit_functional

transformers.logging.set_verbosity_error()


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None, help="Start with warmed-up model weights")
    parser.add_argument("--continue_from_peft", type=str, default=None, help="Continue training with ReLoRA, loading optimizer and scheduler from the checkpoint.")
    parser.add_argument("--restore_optimizer", default=False, action="store_true")

    parser.add_argument("--dataset", type=str, default="c4,en", help="Huggingface dataset name,split (split is optional)")
    parser.add_argument("--streaming_dataset", default=True, type=lambda x: x.lower() == "true")

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--relora", type=int, default=None)
    parser.add_argument("--train_scaling", default=False, action="store_true")
    parser.add_argument("--reset_optimizer_on_relora", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--optimizer_random_pruning", default=0.0, type=float,
                        help="Use random pruning to reduce optimizer matrix internal dimensionality.")
    parser.add_argument("--optimizer_magnitude_pruning", default=0.0, type=float,
                        help="Use magnitude pruning to reduce optimizer matrix internal dimensionality.")
    parser.add_argument("--force_keep_original", default=False, action="store_true",
                        help=("Keep original model parameters even if relora is None. "
                              "Useful for making sure that full-LoRa model is equivalent to model+LoRa."))

    parser.add_argument("--train_ln", default=True, action="store_true")
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "Shampoo"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--cycle_length", type=int, default=None, help="Number of steps per cycle for cosine scheduler")
    parser.add_argument("--restart_warmup_steps", type=int, default=None, help="Number of steps for cosine restarts (only used for cosine_restarts)")
    parser.add_argument("--adjust_step", type=int, default=0, help="Number of steps to adjust the scheduler by. "
                            f"Useful when you want to sync ReLoRA resets with the scheduler for a warmed up model. "
                            f"You need to use it, when your warmup_step % relora_resets != 0")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)

    parser.add_argument("--eval_every", type=int, default=1_000)

    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--distributed_type", type=str, default="ddp", choices=["fsdp", "ddp"])

    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)

    if args.continue_from_peft and not args.restore_optimizer:
        logger.warning("--continue_from_peft is set, but --restore_optimizer is not. "
                       "This means that you will train with the optimizer from the checkpoint, "
                       "but will not save the optimizer state. "
                       "This is probably not what you want.")

    if args.distributed_type == "fsdp" and args.weight_decay > 0:
        raise ValueError("FSDP does not support weight decay yet.")

    args.dataset = args.dataset.split(",")

    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, device, batch_size):
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    _time = time.time()

    val_data = datasets.load_dataset(*args.dataset, split="validation", streaming=True)
    val_data = val_data.shuffle(seed=42)

    val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    ddp_loss_info = torch.zeros(2).to(device)  # [loss, n_tokens]
    tokens_in_batch_info = torch.zeros(1).to(device)

    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")
    for i, batch in enumerate(val_data_mapped.batch(batch_size=batch_size)):
        if i == 0:
            # this way of estiming the number of eval steps
            # is needed to avoid a deadlock when using FSDP 
            tokens_in_batch_info[0] += (batch["input_ids"] != pad_idx).sum().item()
            dist.all_reduce(tokens_in_batch_info, op=dist.ReduceOp.SUM)
            n_eval_iters = int(target_eval_tokens / tokens_in_batch_info[0])

        if i > n_eval_iters: break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss

        tokens_in_batch = (batch["input_ids"] != pad_idx).sum().item()
        ddp_loss_info[0] += loss.detach()
        ddp_loss_info[1] += tokens_in_batch

    # Gather losses across all GPUs
    dist.all_reduce(ddp_loss_info, op=dist.ReduceOp.SUM)
    eval_loss = ddp_loss_info[0] / ddp_loss_info[1]
    evaluated_on_tokens = ddp_loss_info[1].item()
    logger.info(f"Evaluated on {evaluated_on_tokens} tokens, eval loss: {eval_loss:.4f}")

    logger.info(f"Evaluation took {time.time() - _time:.2f} seconds")

    return eval_loss, evaluated_on_tokens


def initialize_fsdp(model, dtype):
    wrapping_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    if dtype in ["bf16", "bfloat16"]:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,  # Gradient communication precision
            buffer_dtype=torch.bfloat16,  # Buffer precision
        )
    elif dtype == "float32":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,  # Gradient communication precision
            buffer_dtype=torch.float32,  # Buffer precision
        )
    else:
        raise ValueError(f"Dtype {dtype} not supported (only float32 and bfloat16 are)")

    model = FSDP(
        model,
        mixed_precision=mixed_precision_policy,
        auto_wrap_policy=wrapping_policy,
    )
    return model


def save_model_ddp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir):
    global_rank = dist.get_rank()
    if global_rank != 0: return

    update_step = training_state_checkpoint["update_step"]

    if global_rank == 0:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    model.module.save_pretrained(save_dir)
    optimizer_checkpoint = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "update_step": update_step,
        "global_step": training_state_checkpoint["global_step"],
        "config": run_config,
        "wandb": wandb.run.dir,
        "dtype": args.dtype,
    }
    torch.save(optimizer_checkpoint, f"{save_dir}/optimizer.pt")

    with open(f"{save_dir}/training_state.json", "w") as f:
        json.dump(training_state_checkpoint, f, indent=4)


def save_model_fsdp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir):
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        global_rank = dist.get_rank()
        update_step = training_state_checkpoint["update_step"]

        if global_rank == 0:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        model.module.save_pretrained(save_dir)

        if global_rank == 0:
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": training_state_checkpoint["global_step"],
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{save_dir}/optimizer.pt")

            with open(f"{save_dir}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)


def save_model(model, *, optimizer, scheduler, training_state_checkpoint, run_config, distributed_type, save_dir):
    """
    Args:
        training_state_checkpoint: dict with keys:
            global_step: int
            update_step: int
            tokens_seen: int
            tokens_seen_before: int
            n_lora_restarts: int
            update_time: float
        run_config: 
    """
    if distributed_type == "ddp":
        save_model_ddp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir)
    elif distributed_type == "fsdp":
        save_model_fsdp(model, optimizer, scheduler, training_state_checkpoint, run_config, save_dir)
    else:
        raise ValueError(f"Unknown distributed type {distributed_type}")


def main(args):
    # seed all
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    using_fsdp = args.distributed_type == "fsdp"

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    print(f"local rank: {args.local_rank}, device: {torch.cuda.current_device()}")

    # assumes that we are using a single node
    dist.init_process_group(
        backend="nccl",
        rank=args.local_rank,
        world_size=torch.cuda.device_count()
    )

    global_rank = dist.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Setting num_training_steps to {args.num_training_steps} based on max_train_tokens")

    # turn off logger
    if global_rank != 0: logger.remove()

    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project="peft_pretraining", tags=args.tags)

    logger.info(f"Using torch.distributed with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # args.dataset can look like this: "[c4, en]", this is why we use the star
    data = datasets.load_dataset(*args.dataset, split="train", streaming=args.streaming_dataset)

    # this seed is hard-coded to guarantee the same order of the examples (for any --seed)
    seed_for_shuffle = 42
    if args.continue_from is not None:
        # add hash of the path to the checkpoint to the seed
        seed_for_shuffle += int(hashlib.sha256(args.continue_from.encode("utf-8")).hexdigest(), 16) % 10**8
    if args.continue_from_peft is not None:
        seed_for_shuffle += int(hashlib.sha256(args.continue_from_peft.encode("utf-8")).hexdigest(), 16) % 10**8

    logger.info(f"Shuffling data with seed {seed_for_shuffle} (should be 42 for the first run and 42 + hash(checkpoint_path) for the runs that continue from a checkpoint)")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    data = datasets.distributed.split_dataset_by_node(
        data, rank=global_rank, world_size=world_size,
    )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")  # !! won't work with sharded models
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)

    params_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.use_peft:
        for p in model.parameters():
            p.requires_grad = False

        need_linear_weight = args.relora is not None or args.force_keep_original
        if args.continue_from is not None:
            need_linear_weight = True

        model = ReLoRaModel(
            model,
            r=args.lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["attn", "mlp"],
            trainable_scaling=args.train_scaling,
            keep_original_weights=args.continue_from is not None,
            lora_only=not need_linear_weight,
        )

        for name, param in model.named_parameters():
            # LLaMa: model.norm, model.layers.input_layernorm, model.layers.post_attention_layernorm
            if args.train_ln and "norm" in name:
                param.requires_grad = True        
            elif "lm_head" in name:
                param.requires_grad = True
            elif "embed_tokens" in name:
                param.requires_grad = True
            elif "bias" in name:
                param.requires_grad = True
            elif "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    if args.continue_from_peft:
        logger.info(f"Loading model from {args.continue_from_peft}")
        checkpoint_path = os.path.join(args.continue_from_peft, "pytorch_model.bin")
        model.wrapped_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
        with open(os.path.join(args.continue_from_peft, "training_state.json")) as f:
            _old_state = json.load(f)
        global_step = _old_state["global_step"]
        update_step = _old_state["update_step"]
        tokens_seen = _old_state["tokens_seen"]
        tokens_seen_before = _old_state["tokens_seen_before"]
        logger.info(f"global_step       : {global_step}")
        logger.info(f"update_step       : {update_step}")
        logger.info(f"tokens_seen       : {tokens_seen}")
        logger.info(f"tokens_seen_before: {tokens_seen_before}")
        logger.info(f"Will train for {args.num_training_steps - update_step} update steps")

    params_after = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params before LoRA: {params_before / 1_000_000:.2f}M")
    logger.info(f"Total params after  LoRA: {params_after / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    if args.use_peft and args.relora is not None:
        if (trainable_after >= trainable_before):
            raise ValueError("Total number of trainable parameters should decrease after applying LoRA with restarts")

    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    if args.distributed_type == "fsdp":
        logger.info("Wrapping model with FSDP")
        assert using_fsdp
        model: Union[FSDP, ReLoRaModel, LlamaForCausalLM] = initialize_fsdp(model, dtype=args.dtype)

    elif args.distributed_type == "ddp":
        logger.info("Wrapping model with DDP")
        model: Union[ReLoRaModel, LlamaForCausalLM] = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p_trainable_params = n_trainable_params / n_total_params

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    trainable_params_names = [name for name, p in model.named_parameters() if p.requires_grad]

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "trainable_params_M": n_trainable_params / 1_000_000,
        "equivalent_params_M": params_before / 1_000_000,
        "percent_trainable_params": p_trainable_params,
        "name_trainable_params": trainable_params_names,
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if args.use_peft:
        logger.warning("PEFT config (all but lora_r) is hardcoded!")
        run_config["peft_config"] = {
            "r": args.lora_r,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["attn", "mlp"],
        }

    if global_rank == 0:
        wandb.config.update(run_config)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    optimizer_state_keys = None
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        optimizer_state_keys = ["exp_avg", "exp_avg_sq"]
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        cycle_length=args.cycle_length,
        restart_warmup_steps=args.restart_warmup_steps,
        adjust_step=args.adjust_step,
    )

    if args.continue_from_peft:
        logger.info("Setting scheduler to the same state as in the checkpoint")
        for _ in range(update_step):
            scheduler.step()
        logger.info(f"Scheduler state restored from {args.continue_from_peft}")
        # current lr
        logger.info(f"Current lr is {optimizer.param_groups[0]['lr']}")

    if args.restore_optimizer:
        _optimizer_dir = args.continue_from_peft or args.continue_from
        optimizer_checkpoint = torch.load(os.path.join(_optimizer_dir, "optimizer.pt"), map_location="cpu")
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        update_step = optimizer_checkpoint["update_step"]
        global_step = optimizer_checkpoint["global_step"]
        logger.info(f"Optimizer and scheduler restored from {_optimizer_dir}")

    # global steps and others are defined above
    n_lora_restarts = 0
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    for batch in dataloader:
        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step
        if global_rank == 0: pbar.update(1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        update_step += 1
        update_time = time.time() - update_time

        if local_step > args.gradient_accumulation and update_step % args.save_every == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "n_lora_restarts": n_lora_restarts,
                "update_time": update_time,
            }
            save_model(
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                training_state_checkpoint=training_state_checkpoint,
                run_config=run_config,
                distributed_type=args.distributed_type,
                save_dir=current_model_directory,
            )

        # restart model after we modify the learning rate, so on the next step after the relora frequency
        can_reset = args.continue_from_peft is not None \
            or (args.relora is not None and local_step * args.gradient_accumulation > args.relora)

        if can_reset and update_step % args.relora == 1:
            logger.info(f"Performing lora reset. Current lr is {optimizer.param_groups[0]['lr']}")
            n_lora_restarts += 1

            if args.distributed_type == "ddp":
                model.module.merge_and_reinit()
            elif args.distributed_type == "fsdp":
                model.apply(merge_and_reinit_functional)
            else:
                raise ValueError(f"Unknown distributed type {args.distributed_type}")

            if args.reset_optimizer_on_relora:
                logger.info("Resetting optimizer states to zeros")
                for p in lora_params:
                    param_state = optimizer.state[p]
                    for key in optimizer_state_keys:
                        param_state[key] = torch.zeros_like(param_state[key])

            # check optimizer learning rate
            # scheduler should provide a new warmup after the reset
            lr = optimizer.param_groups[0]["lr"]
            if lr > 1e-4:
                alert_message = f"Optimizer lr after the reset is large. This can lead to instability. Current lr is {lr}"
                logger.warning(alert_message)
                if global_rank == 0:
                    wandb.alert(
                        title="Learning rate issue",
                        text=alert_message,
                        level=wandb.AlertLevel.WARN,
                    )

            if args.optimizer_random_pruning:
                logger.info(f"Performing random pruning of optimizer states. Pruning {args.optimizer_random_pruning} percent")
                n_zeros = 0
                n_total = 0

                for p in lora_params:
                    param_state = optimizer.state[p]
                    reduction = partial(training_utils.random_pruning, prune_ratio=args.optimizer_random_pruning)
                    for key in optimizer_state_keys:
                        param_state[key] = reduction(param_state[key])

                logger.info(f"Percent of optimizer states zeroed: {n_zeros / (1e-7 + n_total) * 100:.2f}")

            if args.optimizer_magnitude_pruning:
                logger.info(f"Performing magnitude pruning of optimizer states. Pruning {args.optimizer_magnitude_pruning} percent")
                n_zeros = 0
                n_total = 0
                for p in lora_params:
                    param_state = optimizer.state[p]
                    reduction = partial(training_utils.magnitude_pruning, prune_ratio=args.optimizer_magnitude_pruning)
                    for key in optimizer_state_keys:
                        param_state[key] = reduction(param_state[key])

                    n_zeros += (param_state[optimizer_state_keys[0]] == 0).sum()
                    n_total += param_state[optimizer_state_keys[0]].numel()

                logger.info(f"Percent of optimizer states zeroed: {n_zeros / (1e-7 + n_total) * 100:.2f}")

        if can_reset and update_step % args.relora == 2:
            logger.info(f"First step after lora reset lr is {optimizer.param_groups[0]['lr']}")

        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, device, args.batch_size
            )

            if global_rank == 0:
                wandb.log({
                    "final_eval_loss": total_loss,
                    "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        lr = optimizer.param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "n_lora_restarts": n_lora_restarts,
                },
                step=global_step,
            )
            if args.train_scaling:
                all_scaling_factors = []
                for module in model.modules():
                    if isinstance(module, ReLoRaLinear):
                        all_scaling_factors.append(module.scaling.data.item())
                wandb.log({"lora_scaling": torch.tensor(all_scaling_factors)}, step=global_step)
        update_time = time.time()
    else: # for-else statement
        logger.warning("Reached the end of the dataset. Training stopped")

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if not os.path.exists(current_model_directory):
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "n_lora_restarts": n_lora_restarts,
            "update_time": update_time,
        }
        save_model(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state_checkpoint=training_state_checkpoint,
            run_config=run_config,
            distributed_type=args.distributed_type,
            save_dir=args.save_dir,
        )

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, preprocess_batched, pad_idx, device, args.batch_size
    )

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
