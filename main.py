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


import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

import datasets
import wandb

from tqdm import tqdm
from loguru import logger

import peft_pretraining.training_utils as training_utils
from peft_pretraining.relora import ReLoRaModel


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
    parser.add_argument("--reset_optimizer_on_relora", default=True, type=lambda x: x.lower() == "true")

    parser.add_argument("--force_keep_original", default=False, action="store_true",
                        help=("Keep original model parameters even if relora is None. "
                              "Useful for making sure that full-LoRa model is equivalent to model+LoRa."))

    parser.add_argument("--train_ln", default=True, action="store_true")
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
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args(args)

    if not args.train_ln:
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

    return args


def main(args):
    device = args.device or "cuda"
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            args.gradient_accumulation = args.total_batch_size // args.batch_size

    assert args.batch_size * args.gradient_accumulation == args.total_batch_size, \
        "batch_size * gradient_accumulation must be equal to total_batch_size"

    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    dataset_name = "c4"  # switch to "togethercomputer/RedPajama-Data-1T" later
    assert dataset_name == "c4"
    data = datasets.load_dataset("c4", "en", split="train", streaming=True)
    val_data = datasets.load_dataset("c4", "en", split="validation", streaming=True)

    data: datasets.Dataset = data.shuffle(seed=42)
    val_data: datasets.Dataset = val_data.shuffle(seed=42)  # not sure if C4 val set is shuffled originally

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

        keep_original = args.relora is not None or args.force_keep_original
        if args.continue_from is not None:
            keep_original = True

        model = ReLoRaModel(
            model,
            r=args.lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["attn", "mlp"],
            keep_original=keep_original,
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

    # Initialize wandb
    _config = dict(vars(args))
    _config["max_lr"] = _config.pop("lr")  # rename lr to max_lr
    _config_ext = {
        "total_params_M": n_total_params / 1_000_000,
        "trainable_params_M": n_trainable_params / 1_000_000,
        "equivalent_params_M": params_before / 1_000_000,
        "percent_trainable_params": p_trainable_params,
        "name_trainable_params": trainable_params_names,
        "dataset": dataset_name,
        "model": model_config.to_dict(),
        "device": str(device),
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

    wandb.init(project="peft_pretraining", config=_config, tags=args.tags)
    wandb.save(os.path.abspath(__file__), policy="now") # save current script
    pbar = tqdm(total=args.num_training_steps * args.gradient_accumulation - update_step)

    model = model.to(device, dtype=getattr(torch, args.dtype))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
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
        pbar.update(1)
        if update_step > args.num_training_steps:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item()

        loss = model(**batch, labels=labels).loss
        scaled_loss =  loss/ args.gradient_accumulation
        scaled_loss.backward()

        if local_step < 10:
            # kind of logging this out of desperation
            logger.info(f"Loss at step {local_step}: {loss.item()}")
            lr = optimizer.param_groups[0]["lr"]
            wandb.log({"loss": loss.item(), "lr": lr}, step=global_step)


        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        update_step += 1
        update_time = time.time() - update_time

        if update_step % args.save_every == 0:
            logger.info(f"Saving model and optimizer at update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            model.save_pretrained(current_model_directory)
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": _config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "n_lora_restarts": n_lora_restarts,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

        # restart model after we modify the learning rate, so on the next step after the relora frequency
        if args.relora and update_step > args.relora and update_step % args.relora == 1:
            logger.info(f"Performing lora reset. Current lr is {optimizer.param_groups[0]['lr']}")
            n_lora_restarts += 1
            model.merge_and_reinit()

            if args.reset_optimizer_on_relora:
                logger.info("Resetting optimizer states to zeros")
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        param_state = optimizer.state[p]
                        param_state["exp_avg"] = torch.zeros_like(p.data)
                        param_state["exp_avg_sq"] = torch.zeros_like(p.data)

        if args.relora and update_step > args.relora and update_step % args.relora == 2:
            logger.info(f"First step after lora reset lr is {optimizer.param_groups[0]['lr']}")

        lr = scheduler.get_last_lr()[0]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation

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
        update_time = time.time()

    pbar.close()
    logger.info("Training finished. Saving final model and optimizer state.")
    logger.info(f"Saving model at update step {update_step}")
    os.makedirs(args.save_dir, exist_ok=True)
    current_model_directory = f"{args.save_dir}/model_{update_step}"
    model.save_pretrained(current_model_directory)
    optimizer_checkpoint = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "update_step": update_step,
        "global_step": global_step,
        "config": _config,
        "wandb": wandb.run.dir,
        "dtype": args.dtype,
    }
    torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

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

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item()

    total_loss = total_loss / total_batches

    wandb.log({
        "final_eval_loss": total_loss,
        "final_eval_tokens": evaluated_on_tokens,
        },
        step=global_step,
    )
    logger.info(f"Final eval loss: {total_loss}")

    logger.info("Script finished successfully")
    logger.info(f"Final model checkpoint: {current_model_directory}")


if __name__ == "__main__":
    args = parse_args(None)
    main(args)
