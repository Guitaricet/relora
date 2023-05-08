export CUDA_VISIBLE_DEVICES=0
deepspeed deepspeed_main.py \
    --model_config configs/llama_9m.json \
    --use_peft \
    --lr 0.001 \
    --batch_size 240 \
    --gradient_accumulation 2 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype fp16 \

deepspeed deepspeed_main.py \                                                                                                                                                                              ✘ 2
    --model_config configs/llama_20m.json \
    --use_peft \
    --lr 0.001 \
    --batch_size 120 \
    --gradient_accumulation 4 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype fp16 \
    --distributed_port 21531

deepspeed deepspeed_main.py \
    --model_config configs/llama_35m.json \
    --use_peft \
    --lr 0.001 \
    --batch_size 120 \
    --gradient_accumulation 4 \
    --num_training_steps 10000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype fp16 \
    --distributed_port 21532

deepspeed deepspeed_main.py \
    --model_config configs/llama_60m.json \
    --use_peft \
    --lr 0.001 \
    --batch_size 60 \
    --gradient_accumulation 8 \
    --num_training_steps 10000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype fp16 \
    --distributed_port 21533

# exploded
export CUDA_VISIBLE_DEVICES=2,3
deepspeed deepspeed_main.py \
    --model_config configs/llama_130m.json \
    --use_peft \
    --lr 0.001 \
    --batch_size 30 \
    --gradient_accumulation 10 \
    --num_training_steps 2000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype fp16 \
    --distributed_port 21534
