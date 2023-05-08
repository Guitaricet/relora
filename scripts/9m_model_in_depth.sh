python main.py \
    --model_config configs/llama_9m.json \
    --lr 0.001 \
    --batch_size 240 \
    --gradient_accumulation 2 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype bfloat16 \
    --device cuda:0 \
    --lora_r 8 \
    --tags "9m_lora_search"


python main.py \
    --model_config configs/llama_9m.json \
    --lr 0.001 \
    --batch_size 240 \
    --gradient_accumulation 2 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype bfloat16 \
    --device cuda:0 \
    --lora_r 32 \
    --tags "9m_lora_search"

python main.py \
    --model_config configs/llama_9m.json \
    --lr 0.001 \
    --batch_size 240 \
    --gradient_accumulation 2 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype bfloat16 \
    --device cuda:0 \
    --lora_r 64 \
    --tags "9m_lora_search"

python main.py \
    --model_config configs/llama_9m.json \
    --lr 0.001 \
    --batch_size 240 \
    --gradient_accumulation 2 \
    --num_training_steps 5000 \
    --warmup_steps 500 \
    --save_every 2000 \
    --dtype bfloat16 \
    --device cuda:0 \
    --lora_r 256 \
    --tags "9m_lora_search"
