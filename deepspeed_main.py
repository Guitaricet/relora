import os
import time
import argparse
from datetime import datetime
from typing import Union
from pprint import pformat

import torch
import torch.distributed

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

import deepspeed
import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

import peft_pretraining.training_utils as training_utils
from peft_pretraining.relora import ReLoRaModel


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

    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--relora", type=int, default=None)
    parser.add_argument("--force_keep_original", default=False, action="store_true",
                        help=("Keep original model parameters even if relora is None. "
                              "Useful for making sure that full-LoRa model is equivalent to model+LoRa."))

    parser.add_argument("--train_ln", default=True, action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)

    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--distributed_port", type=int, default=29500)
    parser.add_argument("--stage", type=int, default=2, help="DeepSpeed ZeRo optimization stage")

    args = parser.parse_args(args)

    if not args.train_ln:
        logger.error("Are you sure? Not training LN is a bad idea.")
        raise ValueError("Are you sure? Not training LN is a bad idea.")

    if args.save_dir is None:
        # use checkpoints / model name, date and time as save directory
        args.save_dir = f"checkpoints/{args.model_config.split('/')[-1].rstrip('.json')}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    if not args.use_peft:
        # just for more clear hparam logging to wandb
        args.relora = None
        args.lora_r = None
        args.force_keep_original = False

    if args.stage == 3:
        logger.error("Model saving is not impelmented for DeepSpeed ZeRo Stage 3")
        raise NotImplementedError("Model saving is not impelmented for DeepSpeed ZeRo Stage 3")

    if args.tags is not None:
        args.tags = args.tags.split(",")

    return args


def main(args):
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
    args.total_batch_size = args.batch_size * args.gradient_accumulation * world_size
    device = f"cuda:{local_rank}"

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
    model: LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    params_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.use_peft:
        for p in model.parameters():
            p.requires_grad = False

        model = ReLoRaModel(
            model,
            r=args.lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["attn", "mlp"],
            keep_original=args.relora is not None or args.force_keep_original,
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

    # Initialize DeepSpeed
    # Prefer to make config here instead of a file to reduce the number of files
    # Reproducibility is enabled through WandB
    logger.info(args)
    logger.info(args.lr)
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
    
    _config["deepspeed_config"] = deepspeed_config
    logger.info("DeepSpeed config:")
    logger.info(pformat(deepspeed_config))
    if global_rank == 0:
        wandb.init(project="peft_pretraining", config=_config, tags=args.tags)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        pbar = tqdm(total=args.num_training_steps * args.gradient_accumulation)

    # DeepSpeed optimizer uses more memory than PyTorch optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = training_utils.get_scheculer(optimizer, args.scheduler, args.num_training_steps, args.warmup_steps)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=trainable_params,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config_params=deepspeed_config,
    )
    model_engine: Union[ReLoRaModel, deepspeed.DeepSpeedEngine, LlamaForCausalLM]  # help with type checking

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    n_lora_restarts = 0
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()

    # we'll never go through all the data, so no need for epochs
    for batch in data_mapped.batch(batch_size=args.batch_size):
        global_step += 1
        if global_rank == 0: pbar.update(1)
        if global_step > args.num_training_steps * args.gradient_accumulation:
            logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model_engine(**batch, labels=labels).loss
        model_engine.backward(loss)
        model_engine.step()  # deepspeed handles gradient accumulation

        if not model_engine.is_gradient_accumulation_boundary():
            continue

        # The below code is only executed during the update step
        update_step += 1
        update_time = time.time() - update_time

        if update_step % args.save_every == 0:
            logger.info(f"Saving model and optimizer at update step {update_step}")
            os.makedirs(args.save_dir, exist_ok=True)
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            if global_rank == 0:
                model.save_pretrained(current_model_directory)  # won't work with Stage 3

            # I think it should save both model and optimizer this way
            model_engine.save_checkpoint(f"{current_model_directory}/deepspeed_checkpoint")

        if args.relora and update_step % args.relora == 0:
            logger.info("In merge and reinit")
            n_lora_restarts += 1
            model_engine.module.merge_and_reinit()

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
        update_time = time.time()

    logger.info("Training finished. Saving final model and optimizer state.")
    logger.info(f"Saving model and optimizer at update step {update_step}")
    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0:
        pbar.close()
        os.makedirs(args.save_dir, exist_ok=True)
        model.save_pretrained(current_model_directory)  # won't work with Stage 3

    model_engine.save_checkpoint(f"{current_model_directory}/deepspeed_checkpoint")

    # Final evaluation
    logger.info("Running final evaluation")
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
    total_loss = 0.0
    for batch in val_data_mapped.batch(batch_size=args.batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model_engine(**batch, labels=labels).loss
        total_loss += loss.item()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    # gather across all GPUs
    losses = torch.tensor([total_loss], device=device)
    torch.distributed.all_gather(losses, losses)
    losses = losses.cpu().numpy().sum() / world_size
    total_loss = losses

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
