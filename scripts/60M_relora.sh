torchrun --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.001 \
    --warmup_steps 500 \
    --scheduler cosine_restarts \
    --restart_warmup_steps 100 \
    --cycle_length 1000 \
    --batch_size 120 \
    --total_batch_size 480 \
    --num_training_steps 10000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --min_lr_ratio 0.1 \
    --tags "relora 60M May 13" \
    --relora 1000 \
    --reset_optimizer_on_relora False \
    --use_peft \
    --keep_first_opt_rows 32


torchrun --master-port 1234 --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.001 \
    --warmup_steps 500 \
    --scheduler cosine_restarts \
    --restart_warmup_steps 100 \
    --cycle_length 1000 \
    --batch_size 120 \
    --total_batch_size 480 \
    --num_training_steps 10000 \
    --save_every 10000 \
    --dtype bfloat16 \
    --min_lr_ratio 0.1 \
    --tags "relora 60M May 13" \
    --relora 1000 \
    --reset_optimizer_on_relora False \
    --use_peft \
    --keep_first_opt_rows 4 \
    --distributed_port 1234


torchrun --nproc-per-node 2 torchrun_main.py \
    --model_config configs/llama_60m.json \
    --lr 0.001 \
    --scheduler cosine \
    --min_lr_ratio 0.1 \
    --warmup_steps 500 \
    --batch_size 120 \
    --total_batch_size 480 \
    --num_training_steps 10000 \
    --save_every 2000 \
    --dtype bfloat16 \
    --tags "relora 60M May 13,paperV1" \


torchrun --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_40m.json \
    --lr 0.001 \
    --scheduler cosine \
    --min_lr_ratio 0.1 \
    --warmup_steps 500 \
    --batch_size 120 \
    --total_batch_size 480 \
    --num_training_steps 10000 \
    --save_every 2000 \
    --dtype bfloat16 \
    --tags "relora 60M May 13,paperV1" \


torchrun --master-port 1234 --nproc-per-node 4 torchrun_main.py \
    --model_config configs/llama_60m.json --lr 0.001 --warmup_steps 500 --scheduler cosine --batch_size 120 --total_batch_size 480 --num_training_steps 10000 --save_every 10000 --dtype bfloat16 --min_lr_ratio 0.1 --tags "relora 60M May 13" --relora 1000 --reset_optimizer_on_relora False --use_peft