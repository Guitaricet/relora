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


## Citation

```
@misc{lialin2023stack,
    title={Stack More Layers Differently: High-Rank Training Through Low-Rank Updates},
    author={Vladislav Lialin and Namrata Shivagunde and Sherin Muckatira and Anna Rumshisky},
    year={2023},
    eprint={2307.05695},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
