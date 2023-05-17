# PEFT Pretraining

## Setup

If you are setting the repository on a **new instance**, all you need to do is:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash miniconda.sh -b
/root/miniconda3/bin/conda init

bash  # to enable conda

cd peft_pretraining
pip install -r requirements.txt

wandb login
```

## Usage

Train language model with PEFT
```bash
python main.py --model_config configs/llama_1b.json --use_peft --device cuda:0 --lr 0.0005 --batch_size 16
```

Train without PEFT
```bash
python main.py --model_config configs/llama_1b.json --device cuda:0 --lr 0.0005 --batch_size 16
```

Train a larger model
```bash
python main.py \
    --model_config configs/llama_3b.json \
    --use_peft --train_ln \
    --device cuda:0 \
    --lr 0.0005 \
    --batch_size 16 \
    --gradient_accumulation 4 \
    --num_training_steps 50000
```

## Note on batch sizes

To minimize the pain with multi-GPU setups, we recommend to avoid using `--gradient_accumulation` option directly. Instead specify `--total_batch_size` and allow the script to figure out the gradient accumulation option based on `--batch_size` and the number of GPUs used.

## Relora

Relora integrates existing LoRA parameters into the main network and resets them.
In principle, such approach can be more flexible than LoRA, but you need to be careful with

1. Optimizer states
2. Learning rate schedule during and right after the reset
3. How frequently you reset

Reset frequency is determined by `--relora` parameter (in the number of update steps, not global steps).
Optimizer reset options are: 
```
"--reset_optimizer_on_relora", default=True, type=lambda x: x.lower() == "true"

"--svd_optimizer_on_relora", default=0, type=int,
help="Instead of resetting optimizer, take top-k singular values of the optimizer matrix."

"--keep_first_opt_rows", default=0, type=int,
help="Instead of resetting optimizer, zero all but --keep_first_opt_rows rows in matricies"
```

We curently support linear and cosine decay learning rate schedulers it is defined by 
Cosine schedule also supports cyclical mode that repeat the warmup and decay every `--cycle_length` update steps.

## Warm starts

You can start LoRa from a partially trained checkpoint. To do that, provide `--continue_from` option. For example:

```
torchrun torchrun_main.py ... <other options> .. --continue_from checkpoints/llama_1b-2023-05-05-20-12-43/model_1000
```

## Distributed training

We support single-node distributed training. For it you can use deepspeed or torchrun.

We had some issues with deepspeed and even though it has some nice extra features, we recommend to use torchrun, especially for relora experiments that explode with deepspeed for the reason we coudn't decifer.

Example of using torchrun
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