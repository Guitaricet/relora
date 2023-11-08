Some script to check that the most common training reigmes work.

```
torchrun --nproc-per-node 2 torchrun_main.py \
    --dataset_path preprocessed_data/wikitext_wikitext-2-v1_EleutherAI_pythia-1.4b_512 \
    --model_name_or_path EleutherAI/pythia-1.4b \
    --use_peft \
    --relora 10 \
    --model_revision step1000 \
    --batch_size 4 \
    --total_batch_size 96 \
    --lr 5e-4 \
    --max_length 512 \
    --eval_every 20 \
    --save_every 20 \
    --num_training_steps 40 \
    --distributed_type ddp \
    --optimizer adam_zero \
    --tags debug


torchrun --nproc-per-node 2 torchrun_main.py \
    --dataset_path preprocessed_data/wikitext_wikitext-2-v1_EleutherAI_pythia-1.4b_512 \
    --model_name_or_path EleutherAI/pythia-1.4b \
    --model_revision step1000 \
    --batch_size 6 \
    --total_batch_size 96 \
    --lr 5e-4 \
    --max_length 512 \
    --eval_every 2 \
    --save_every 10 \
    --num_training_steps 20 \
    --distributed_type ddp \
    --tags debug,fsdp_debug


torchrun --nproc-per-node 2 torchrun_main.py \
    --dataset_path preprocessed_data/wikitext_wikitext-2-v1_t5-base_512 \
    --model_config configs/llama_250m.json \
    --batch_size 24 \
    --total_batch_size 96 \
    --lr 5e-4 \
    --max_length 512 \
    --eval_every 2 \
    --save_every 10 \
    --num_training_steps 20 \
    --distributed_type ddp \
    --tags debug,fsdp_debug


torchrun --nproc-per-node 2 torchrun_main.py \
    --dataset_path preprocessed_data/wikitext_wikitext-2-v1_t5-base_512 \
    --model_config configs/llama_250m.json \
    --batch_size 24 \
    --total_batch_size 96 \
    --lr 5e-4 \
    --max_length 512 \
    --eval_every 2 \
    --save_every 10 \
    --num_training_steps 20 \
    --distributed_type fsdp \
    --tags debug,fsdp_debug


torchrun --nproc-per-node 2 torchrun_main.py \
    --dataset_path preprocessed_data/wikitext_wikitext-2-v1_gpt2_512 \
    --model_config configs/llama_250m_50K.json \
    --batch_size 24 \
    --total_batch_size 96 \
    --lr 5e-4 \
    --max_length 512 \
    --eval_every 2 \
    --save_every 10 \
    --num_training_steps 20 \
    --distributed_type ddp \
    --dtype float32 \
    --tags debug,fsdp_debug


torchrun --nproc-per-node 2 torchrun_main.py \
    --model_config configs/llama_250m.json \
    --batch_size 24 \
    --total_batch_size 96 \
    --lr 5e-4 \
    --max_length 512 \
    --eval_every 2 \
    --save_every 10 \
    --num_training_steps 20000 \
    --distributed_type fsdp \
    --tags debug,fsdp_debug


torchrun --nproc-per-node 2 torchrun_main.py \
    --model_config configs/llama_250m.json \
    --batch_size 24 \
    --total_batch_size 96 \
    --lr 5e-4 \
    --max_length 512 \
    --eval_every 2 \
    --save_every 10 \
    --num_training_steps 20000 \
    --distributed_type fsdp \
    --tags debug,fsdp_debug


torchrun --nproc-per-node 2 torchrun_main.py \
    --model_config configs/llama_250m.json \
    --batch_size 24 \
    --total_batch_size 96 \
    --lr 1e-3 \
    --max_length 512 \
    --use_peft \
    --relora 10 \
    --cycle_length 10 \
    --restart_warmup_steps 5 \
    --scheduler cosine_restarts \
    --warmup_steps 5 \
    --reset_optimizer_on_relora False \
    --optimizer_magnitude_pruning 0.9 \
    --num_training_steps 20000 \
    --save_every 5000 \
    --eval_every 5000 \
    --warmed_up_model checkpoints/llama_250m-2023-06-09-11-29-56/model_5000 \
    --distributed_type fsdp \
    --tags debug,fsdp_debug


torchrun --nproc-per-node 2 torchrun_main.py \
    --model_config configs/llama_250m.json \
    --batch_size 24 \
    --total_batch_size 96 \
    --lr 1e-3 \
    --max_length 512 \
    --use_peft \
    --relora 10 \
    --cycle_length 10 \
    --restart_warmup_steps 5 \
    --scheduler cosine_restarts \
    --warmup_steps 5 \
    --reset_optimizer_on_relora False \
    --optimizer_magnitude_pruning 0.9 \
    --num_training_steps 20000 \
    --save_every 5000 \
    --eval_every 5000 \
    --warmed_up_model checkpoints/llama_250m-2023-06-09-11-29-56/model_5000 \
    --distributed_type fsdp \
    --tags debug,fsdp_debug

```
