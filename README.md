# PEFT Pretraining
> Official code for Stack More Layers Differently: High-Rank Training Through Low-Rank Updates https://arxiv.org/abs/2307.05695

## Setup

All requirements are listed in `requirements.txt` and kept up-to-date.

```bash
cd peft_pretraining
pip install -r requirements.txt
```

## Usage

To train a model using ReLoRA, first, perform a warmup through regular training.

Train language model with PEFT
```bash
torchrun --nproc-per-node <N_GPUS> torchrun_main.py \
    --model_config configs/llama_250m.json \
    --batch_size 24 \
    --total_batch_size 1152 \
    --lr 5e-4 \
    --max_length 512 \
    --tags warm_start_250M \
    --save_every 1000 \
    --num_training_steps 20000
```

> **Reproducibility note:** The way we ran the experiments in the paper was by specifying full num_training_steps, including both the warmup and the ReLoRA training, and stopping it after the desired number of steps was completed. Providing only the number of training steps should work too. The only difference will be the LR schedule during the warmup period.

When you have a warmed-up network checkpoint, run the script with ReLoRA enabled. Note that we use a larger LR during the ReLoRA stage.

Train without PEFT
```bash
torchrun --nproc-per-node <N_GPUS> torchrun_main.py \
    --model_config configs/llama_250m.json \
    --batch_size 24 \
    --total_batch_size 1152 \
    --lr 1e-3 \
    --max_length 512 \
    --use_peft \
    --relora 5000 \
    --cycle_length 5000 \
    --restart_warmup_steps 100 \
    --scheduler cosine_restarts \
    --warmup_steps 500 \
    --reset_optimizer_on_relora True \
    --num_training_steps 20000 \
    --save_every 5000 \
    --eval_every 5000 \
    --continue_from checkpoints/llama_250m-2023-06-09-11-29-56/model_5000 \
    --tags relora_250M
```



## Note on batch sizes

To minimize the pain with multi-GPU setups, we recommend avoiding using `--gradient_accumulation` option directly. Instead, specify `--total_batch_size` and allow the script to figure out the gradient accumulation option based on `--batch_size` and the number of GPUs used.

## Relora

Relora integrates existing LoRA parameters into the main network and resets them.
In principle, such an approach can be more flexible than LoRA, but you need to be careful with

1. Optimizer states
2. Learning rate schedule during and right after the reset
3. How frequently you reset

Reset frequency is determined by `--relora` parameter (in the number of update steps, not global steps).
Optimizer reset options are: 
```
"--reset_optimizer_on_relora", default=True, type=lambda x: x.lower() == "true"
"--optimizer_random_pruning", default=False, type=float
"--optimizer_magnitude_pruning", default=False, type=float
```

We found that using `--optimizer_magnitude_pruning 0.9` or plain `--reset_optimizer_on_relora` usually performs well.
Note that `--reset_optimizer_on_relora is True by default` and you need to provide `--reset_optimizer_on_relora False --optimizer_magnitude_pruning 0.9` if you want to do magnitude pruning.

ReLoRA currently only supports cosine decay learning rate scheduler.
Specifically `cosine_restarts` that works in cyclical mode that repeats the warmup every `--cycle_length` update steps.

## Warm starts

You can start LoRa from a partially trained checkpoint. To do that, provide `--continue_from` option. For example:

```
torchrun torchrun_main.py ... <other options> .. --continue_from checkpoints/llama_1b-2023-05-05-20-12-43/model_1000
```

## Distributed training

We support single-node distributed training using vanilla PyTorch DDP.
| `main.py` script does not have all features required for relora and will be deleted soon. We recommend to use `torchrun --nproc-per-node 1` for a single-GPU training.

An example of using torchrun
```bash
torchrun --nproc-per-node 8 torchrun_main.py \
    --model_config configs/llama_35m.json \
    --use_peft \
    --lora_r 128 \
    --relora 500 \
    --cycle_length 500 \
    --warmup_steps 250 \
    --reset_optimizer_on_relora False \
    --lr 0.001 \
    --batch_size 60 \
    --total_batch_size 480 \
    --num_training_steps 5000 \
    --save_every 5000 \
    --dtype bfloat16 \
    --tags relora_debug,example
```

Where `--nproc-per-node` is the nubmer of GPUs you are using.


### More usage examples

```
torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_130m.json --lr 0.001 --warmup_steps 500 --scheduler cosine_restarts --restart_warmup_steps 100 --cycle_length 5000 --relora 5000 --use_peft --batch_size 75 --total_batch_size 600 --num_training_steps 20000 --save_every 20000 --dtype bfloat16 --min_lr_ratio 0.1 --tags "relora 130M May 14" --reset_optimizer_on_relora False --optimizer_magnitude_pruning 0.7 --continue_from checkpoints/llama_130m-2023-05-14-19-54-05/model_5000/

torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_130m.json --lr 0.001 --warmup_steps 500 --scheduler cosine_restarts --restart_warmup_steps 100 --cycle_length 5000 --relora 5000 --use_peft --batch_size 75 --total_batch_size 600 --num_training_steps 20000 --save_every 20000 --dtype bfloat16 --min_lr_ratio 0.1 --tags "relora 130M May 14" --reset_optimizer_on_relora False --optimizer_magnitude_pruning 0.5 --continue_from checkpoints/llama_130m-2023-05-14-19-54-05/model_5000/


torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_60m.json --lr 0.001 --warmup_steps 500 --scheduler cosine_restarts --restart_warmup_steps 50 --cycle_length 2000 --relora 2000 --batch_size 60 --total_batch_size 480 --num_training_steps 10000 --save_every 20000 --dtype bfloat16 --min_lr_ratio 0.1 --tags "relora 60M May 13" --reset_optimizer_on_relora False --use_peft --optimizer_magnitude_pruning 0.9

torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_250m.json --lr 0.001 --warmup_steps 500 --scheduler cosine_restarts --restart_warmup_steps 50 --cycle_length 2000 --relora 2000 --batch_size 60 --total_batch_size 480 --num_training_steps 10000 --save_every 20000 --dtype bfloat16 --min_lr_ratio 0.1 --tags "relora 60M May 13" --reset_optimizer_on_relora False --use_peft --optimizer_magnitude_pruning 0.9

torchrun --nproc-per-node 8 torchrun_main.py \
    --model_config configs/llama_250m.json --max_length 512 \
    --use_peft --relora 5000 --cycle_length 5000 --restart_warmup_steps 100 \
    --lr 5e-4 --batch_size 12 --total_batch_size 1152 --scheduler cosine_restarts --warmup_steps 500 \
    --reset_optimizer_on_relora False --optimizer_magnitude_pruning 0.99 \
    --num_training_steps 20000 --save_every 5000 --eval_every 5000 \
    --continue_from checkpoints/llama_250m-2023-05-13-13-56-33/model_5000 \
    --tags warm_start_250M,paperV1

torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_100m.json --batch_size 48 --total_batch_size 1152 --lr 5e-4 --max_length 512 --tags warm_start_250M --save_every 5000 --eval_every 5000 --num_training_steps 20000

torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_1b.json --batch_size 6 --total_batch_size 1920 --lr 2e-4 --max_length 512 --warmup_steps 1000 --num_training_steps 20000 --save_every 1000 --eval_every 1000 --tags "1B_May15"

torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_250m.json --batch_size 24 --use_peft --total_batch_size 1152 --lr 5e-4 --max_length 512 --tags paperV1 --save_every 5000 --eval_every 5000 --num_training_steps 20000

torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_130m.json --lr 0.001 --warmup_steps 500 --scheduler cosine_restarts --restart_warmup_steps 100 --cycle_length 5000 --relora 5000 --use_peft --batch_size 75 --total_batch_size 600 --num_training_steps 20000 --save_every 2500 --eval_every 2500 --dtype bfloat16 --min_lr_ratio 0.1 --tags "relora 130M May 16" --reset_optimizer_on_relora False --optimizer_magnitude_pruning 0.99 --continue_from checkpoints/llama_130m-2023-05-14-19-54-05/model_5000/

# 350 full
torchrun_main.py --model_config configs/llama_350m.json --batch_size 16 --total_batch_size 1152 --lr 5e-4 --max_length 512 --warmup_steps 1000 --num_training_steps 20000 --save_every 1000 --eval_every 1000 --tags 350M_May15,paperV1,restart --restore_optimizer --continue_from checkpoints/llama_350m-2023-05-16-14-41-16/model_9000

# 350 relora
torchrun --nproc-per-node 8 torchrun_main.py --model_config configs/llama_350m.json --batch_size 16 --total_batch_size 1152 --max_length 512 --lr 5e-4 --warmup_steps 1000 --scheduler cosine_restarts --restart_warmup_steps 100 --cycle_length 5000 --relora 5000 --use_peft --num_training_steps 20000 --save_every 5000 --dtype bfloat16 --min_lr_ratio 0.1 --tags "relora 350M May 17" --reset_optimizer_on_relora False --optimizer_magnitude_pruning 0.7 --continue_from checkpoints/llama_350m-2023-05-16-14-41-16/model_5000/

torchrun --nproc-per-node 8 torchrun_main.py --use_peft --model_config configs/llama_130m.json --lr 0.001 --scheduler cosine --warmup_steps 500 --batch_size 75 --total_batch_size 600 --num_training_steps 20000 --save_every 5000 --eval_every 5000 --dtype bfloat16 --relora 5000 --reset_optimizer_on_relora False --optimizer_magnitude_pruning 0.99  --tags paperV1,ablation_130M
```
