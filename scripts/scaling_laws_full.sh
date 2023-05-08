export CUDA_VISIBLE_DEVICES=0
python main.py \
    --model_config configs/llama_9m.json \
    --lr 0.001 \
    --batch_size 240 \
    --gradient_accumulation 2 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype bfloat16 \

python main.py \
    --model_config configs/llama_20m.json \
    --lr 0.001 \
    --batch_size 120 \
    --gradient_accumulation 4 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype bfloat16 \

python main.py \
    --model_config configs/llama_35m.json \
    --lr 0.001 \
    --batch_size 60 \
    --gradient_accumulation 8 \
    --num_training_steps 10000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype bfloat16 \

python main.py \
    --model_config configs/llama_60m.json \
    --lr 0.001 \
    --batch_size 120 \
    --gradient_accumulation 4 \
    --num_training_steps 10000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype bfloat16 \


deepspeed deepspeed_main.py \
    --model_config configs/llama_130m.json \
    --lr 0.001 \
    --batch_size 60 \
    --gradient_accumulation 2 \
    --num_training_steps 20000 \
    --warmup_steps 500 \
    --save_every 4000 \
    --dtype bfloat16 \
    --distributed_port 21538
