import argparse

import torch

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaModel
from peft import get_peft_model, TaskType

import datasets
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining.gpt2 import GPT2LMHeadModel
from peft_pretraining.relora import ReLoRaModel


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, default="peft_pretraining/llama_small.json")
    parser.add_argument("--num_layers", type=int, default=None)

    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--use_peft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--relora", type=int, default=None)

    parser.add_argument("--train_ln", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1_000)

    parser.add_argument("--num_training_steps", type=int, default=10_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")

    args = parser.parse_args(args)
    return args


def main(args):
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    # use f-string formatting to align the arguments
    for k, v in vars(args).items():
        logger.info(f"{k:20} {v}")
    logger.info("*" * 40)

    data = datasets.load_dataset("c4", "en", split="train", streaming=True)
    data = data.shuffle(seed=42)

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

    data_mapped.batch = lambda batch_size: batch_fn(data_mapped, batch_size)

    device = args.device or "cuda"

    model_config = AutoConfig.from_pretrained(args.model_config)

    if args.num_layers is not None:
        model_config.n_layer = args.num_layers

    # model = GPT2LMHeadModel.from_config(model_config)
    model = AutoModelForCausalLM.from_config(model_config)

    params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.use_peft:
        model = ReLoRaModel(
            model,
            r=args.lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["attn", "mlp"],
        )

        for name, param in model.named_parameters():
            # LLaMa
            # model.norm, model.layers.input_layernorm, model.layers.post_attention_layernorm
            if args.train_ln and "norm" in name:
                param.requires_grad = True        
            if "lm_head" in name:
                param.requires_grad = True
            if "embed_tokens" in name:
                param.requires_grad = True
            # LLaMa uses rotary embeddings and they are not trainable

            # GPT2
            # if args.train_ln and "ln_" in name:
            #     param.requires_grad = True
            # if "lm_head" in name:
            #     param.requires_grad = True
            # if "wte" in name:
            #     param.requires_grad = True
            # if "wpe" in name:
            #     param.requires_grad = True

    params_after = sum(p.numel() for p in model.parameters())
    if args.use_peft:
        assert params_after > params_before

    # print params and trainable params
    print(model)
    logger.info(f"Total params before LoRA: {params_before / 1_000_000:.2f}M")
    logger.info(f"Total params after  LoRA: {params_after / 1_000_000:.2f}M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M")

    # print all trainable modules
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.requires_grad:
            logger.info(f"{name:40} {module.weight.shape}")

    model = model.to(device, dtype=getattr(torch, args.dtype))

    n_total_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p_trainable_params = n_trainable_params / n_total_params

    trainable_params = (p for p in model.parameters() if p.requires_grad)
    trainable_params_names = [name for name, p in model.named_parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_training_steps,
    )

    _config = {
        "using_peft": args.use_peft,
        "layer_norm_trainable": args.train_ln,
        "total_params": n_total_params,
        "trainable_params": n_trainable_params,
        "percent_trainable_params": p_trainable_params,
        "name_trainable_params": trainable_params_names,
        "dataset": "c4",
        "batch_size": args.batch_size,
        "max_lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "max_length": args.max_length,
        "model": model_config.to_dict(),
        "scheduler": "linear",
        "device": str(device),
    }
    if args.use_peft:
        logger.warning("PEFT config is hardcoded!")
        _config["peft_config"] = {
            "r": args.lora_r,
            "alpha": 32,
            "dropout": 0.1,
            "target_modules": ["attn", "mlp"],
        }

    wandb.init(project="peft_pretraining", config=_config)
    pbar = tqdm(total=args.num_training_steps)

    global_step = 0
    for epoch in range(1):  # we'll probably never go through all the data
        data_mapped.set_epoch(epoch)
        for batch in data_mapped.batch(batch_size=args.batch_size):
            global_step += 1
            pbar.update(1)
            if global_step > args.num_training_steps:
                break

            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == 0] = -100

            loss = model(**batch, labels=labels).loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.relora and global_step % args.relora == 0:
                print("In merge and reinit")
                model.merge_and_reinit()

            lr = scheduler.get_last_lr()[0]
            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                },
                step=global_step,
            )

    pbar.close()
    logger.info("Training finished")


if __name__ == "__main__":
    args = parse_args(None)
    main(args)
