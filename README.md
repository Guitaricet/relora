# PEFT Pretraining

## Setup

If you are setting the repository on a **new instance**, all you need to do is:

```bash
sudo apt-get install g++ -y
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash miniconda.sh -b
/root/miniconda3/bin/conda init

bash  # to enable conda

git clone https://github.com/guitaricet/peft_pretraining
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
