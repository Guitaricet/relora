import os
import time
import json
import random
import argparse
from datetime import datetime
from typing import Union
from pprint import pformat

import numpy as np

import torch
import torch.nn as nn
import torch.distributed

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from lion_pytorch import Lion

import deepspeed
import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

import peft_pretraining.training_utils as training_utils
from peft_pretraining.relora import ReLoRaModel, ReLoRaLinear


# Use bfloat16 instead of fp16 if possible
DEFAULT_FP16_CONFIG = {
    "enabled": True,
    "auto_cast": False,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
}

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--continue_from", type=str, default=None)

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--relora", type=int, default=None)
    parser.add_argument("--train_scaling", default=False, action="store_true")
    parser.add_argument("--reset_optimizer_on_relora", default=True, type=lambda x: x.lower() == "true")
    parser.add_argument("--force_keep_original", default=False, action="store_true",
                        help=("Keep original model parameters even if relora is None. "
                              "Useful for making sure that full-LoRa model is equivalent to model+LoRa."))

    parser.add_argument("--train_ln", default=True, action="store_true")
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "Lion"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--cycle_length", type=int, default=None, help="Number of steps per cycle for cosine scheduler")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)

    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float16")

    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--distributed_port", type=int, default=29500)
    parser.add_argument("--stage", type=int, default=2, help="DeepSpeed ZeRo optimization stage")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(args)

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

    if args.stage == 3:
        logger.error("Model saving is not impelmented for DeepSpeed ZeRo Stage 3")
        raise NotImplementedError("Model saving is not impelmented for DeepSpeed ZeRo Stage 3")

    if args.total_batch_size is None:
        args.gradient_accumulation = args.gradient_accumulation or 1
        args.total_batch_size = args.batch_size * args.gradient_accumulation

    assert args.total_batch_size % args.batch_size == 0, "total_batch_size must be divisible by batch_size"

    if args.max_train_tokens is not None:
        args.num_training_steps = args.max_train_tokens // args.total_batch_size
        logger.info(f"Training for {args.num_training_steps} update steps")

    if args.continue_from is not None:
        assert os.path.exists(args.continue_from), f"--continue_from={args.continue_from} does not exist"

    return args


def main(args):
    # seed all
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])  # support torchrun

    # assumes that we are using a single node
    deepspeed.init_distributed(
        distributed_port=args.distributed_port,
        init_method=f"tcp://127.0.0.1:{args.distributed_port}",
        rank=args.local_rank,
        world_size=torch.cuda.device_count(),
    )

    global_rank = torch.distributed.get_rank()
    local_rank = global_rank % torch.cuda.device_count()
    world_size = torch.distributed.get_world_size()
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()

    logger.info(f"Using DeepSpeed with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    dataset_name = "c4"  # switch to "togethercomputer/RedPajama-Data-1T" later
    assert dataset_name == "c4"
    data = datasets.load_dataset("c4", "en", split="train", streaming=True)
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True)

    # this seed is hard-coded to guarantee the same order of the examples (for any --seed)
    data: datasets.Dataset = data.shuffle(seed=42)
    val_data: datasets.Dataset = val_data.shuffle(seed=42)  # not sure if C4 val set is shuffled originally
    data = datasets.distributed.split_dataset_by_node(
        data, rank=global_rank, world_size=world_size,
    )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    data_mapped = data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )

    # Very inefficient, but we are bottlenecked by the model anyway
    # as long as the model is large enough
    data_mapped.batch = lambda batch_size: training_utils.batch_fn(data_mapped, batch_size)

    model_config = AutoConfig.from_pretrained(args.model_config)
    model: Union[LlamaForCausalLM, nn.Module] = AutoModelForCausalLM.from_config(model_config)
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
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

    params_after = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params before LoRA: {params_before / 1_000_000:.2f}M")
    logger.info(f"Total params after  LoRA: {params_after / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    if args.save_dir:
        logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    if args.use_peft and args.relora is not None:
        if (params_after <= params_before):
            raise ValueError("Total number of parameters should increase after applying LoRA with restarts")
        
        if (trainable_after >= trainable_before):
            raise ValueError("Total number of trainable parameters should decrease after applying LoRA with restarts")

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p_trainable_params = n_trainable_params / n_total_params

    trainable_params = (p for p in model.parameters() if p.requires_grad)
    trainable_params_names = [name for name, p in model.named_parameters() if p.requires_grad]

    # Initialize DeepSpeed
    # Prefer to make config here instead of a file to reduce the number of files
    # Reproducibility is enabled through WandB
    deepspeed_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "zero_optimization": {
            "stage": args.stage,
        }
    }
    if args.dtype == "bfloat16":
        deepspeed_config["bf16"] = {"enabled": True}
    elif args.dtype in ["fp16", "float16"]:
        deepspeed_config["fp16"] = DEFAULT_FP16_CONFIG

    logger.info("DeepSpeed config:")
    logger.info(pformat(deepspeed_config))
    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=trainable_params,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config_params=deepspeed_config,
    )
    model: Union[ReLoRaModel, deepspeed.DeepSpeedEngine, LlamaForCausalLM]  # help with type checking

    trainable_params = (p for p in model.parameters() if p.requires_grad)  # update trainable params after initializing deepspeed

    # Initialize wandb
    _config = dict(vars(args))
    _config["max_lr"] = _config.pop("lr")  # rename lr to max_lr to avoid conflicts with scheduler
    _config_ext = {
        "total_params_M": n_total_params / 1_000_000,
        "trainable_params_M": n_trainable_params / 1_000_000,
        "equivalent_params_M": params_before / 1_000_000,
        "percent_trainable_params": p_trainable_params,
        "name_trainable_params": trainable_params_names,
        "dataset": dataset_name,
        "model": model_config.to_dict(),
        "device": str(device),
        "deepspeed_config": deepspeed_config,
    }
    _config.update(_config_ext)

    if args.use_peft:
        logger.warning("PEFT config (all but lora_r) is hardcoded!")
        _config["peft_config"] = {
            "r": args.lora_r,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["attn", "mlp"],
        }

    if global_rank == 0:
        wandb.init(project="peft_pretraining", config=_config, tags=args.tags)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        pbar = tqdm(total=args.num_training_steps * args.gradient_accumulation - update_step)

    # DeepSpeed optimizer uses more memory than PyTorch optimizer
    if args.optimizer.lower() == "lion":
        optimizer = Lion(trainable_params, lr=args.lr, weight_decay=args.weight_decay, use_triton=True)
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        num_training_steps=args.num_training_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        cycle_length=args.cycle_length,
    )

    # global steps and others are defined above
    n_lora_restarts = 0
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # we'll never go through all the data, so no need for epochs
    for batch in data_mapped.batch(batch_size=args.batch_size):
        global_step += 1
        local_step += 1

        if global_rank == 0: pbar.update(1)
        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        model.backward(loss)
        model.step()  # deepspeed handles gradient accumulation

        if local_step < 10 and global_rank == 0:
            # kind of logging this out of desperation
            logger.info(f"Loss at step {local_step}: {loss.item()}")
            lr = optimizer.param_groups[0]["lr"]
            wandb.log({"loss": loss.item(), "lr": lr}, step=global_step)

        if not model.is_gradient_accumulation_boundary():
            continue

        # The below code is only executed during the update step
        update_step += 1
        update_time = time.time() - update_time

        if update_step % args.save_every == 0:
            logger.info(f"Saving model and optimizer at update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            if global_rank == 0:
                model.module.save_pretrained(current_model_directory)  # won't work with Stage 3

            # I think it should save both model and optimizer this way
            model.save_checkpoint(f"{current_model_directory}/deepspeed_checkpoint")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "n_lora_restarts": n_lora_restarts,
                "update_time": update_time,
            }
            if global_rank == 0:
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)

        # restart model after we modify the learning rate, so on the next step after the relora frequency
        if args.relora and update_step > args.relora and update_step % args.relora == 1:
            logger.info(f"Performing lora reset. Current lr is {optimizer.param_groups[0]['lr']}")
            n_lora_restarts += 1
            model.module.merge_and_reinit()

            if args.reset_optimizer_on_relora:
                logger.info("Resetting optimizer states to zeros")
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        param_state = optimizer.state[p]
                        param_state["exp_avg"] = torch.zeros_like(p.data)
                        param_state["exp_avg_sq"] = torch.zeros_like(p.data)

        if args.relora and update_step > args.relora and update_step % args.relora == 2:
            logger.info(f"First step after lora reset lr is {optimizer.param_groups[0]['lr']}")

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

    logger.info("Training finished. Saving final model and optimizer state.")
    logger.info(f"Saving model and optimizer at update step {update_step}")
    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0:
        pbar.close()
        os.makedirs(args.save_dir, exist_ok=True)
        model.module.save_pretrained(current_model_directory)  # won't work with Stage 3

    model.save_checkpoint(f"{current_model_directory}/deepspeed_checkpoint")

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    val_data = datasets.distributed.split_dataset_by_node(
        val_data, rank=global_rank, world_size=world_size,
    )
    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    with torch.no_grad():
        for batch in val_data_mapped.batch(batch_size=args.batch_size):
            if evaluated_on_tokens > target_eval_tokens:
                break
            total_batches += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            loss = model(**batch, labels=labels).loss
            total_loss += loss.detach()

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # gather across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

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
    args = parse_args(None)
    main(args)
