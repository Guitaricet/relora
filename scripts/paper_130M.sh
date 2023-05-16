torchrun --master-port 1234 --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --lr 0.001 \
    --scheduler cosine \
    --min_lr_ratio 0.1 \
    --warmup_steps 500 \
    --batch_size 75 \
    --total_batch_size 600 \
    --num_training_steps 20000 \
    --save_every 5000 \
    --dtype bfloat16 \
    --tags "paperV1"

torchrun --master-port 1235 --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --use_peft \
    --relora 2000 \
    --cycle_length 2000 \
    --restart_warmup_steps 100 \
    --reset_optimizer_on_relora False \
    --optimizer_random_pruning 0.9 \
    --lr 0.001 \
    --scheduler cosine \
    --min_lr_ratio 0.1 \
    --warmup_steps 500 \
    --batch_size 50 \
    --total_batch_size 600 \
    --num_training_steps 20000 \
    --save_every 5000 \
    --dtype bfloat16 \
    --tags "relora 130M May13"

torchrun --master-port 1236 --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --use_peft \
    --relora 2000 \
    --cycle_length 2000 \
    --restart_warmup_steps 100 \
    --reset_optimizer_on_relora False \
    --optimizer_magnitude_pruning 0.9 \
    --lr 0.001 \
    --scheduler cosine \
    --min_lr_ratio 0.1 \
    --warmup_steps 500 \
    --batch_size 50 \
    --total_batch_size 600 \
    --num_training_steps 20000 \
    --save_every 5000 \
    --dtype bfloat16 \
    --tags "relora 130M May13"


torchrun --master-port 1236 --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_71m.json \
    --lr 0.001 \
    --scheduler cosine \
    --min_lr_ratio 0.1 \
    --warmup_steps 500 \
    --batch_size 75 \
    --total_batch_size 600 \
    --num_training_steps 20000 \
    --save_every 5000 \
    --dtype bfloat16 \
    --tags "relora 130M May13"
