# PEFT Pretraining

## Setup

```bash
pip install -r requirements.txt
```

## Usage

It's important to provide `--train_ln` when training with PEFT.


Train language model with PEFT
```bash
python main.py --model_config peft_pretraining/llama_1b.json --use_peft --train_ln --device cuda:0 --lr 0.0005 --batch_size 16
```

Train without PEFT
```bash
python main.py --model_config peft_pretraining/llama_1b.json --device cuda:0 --lr 0.0005 --batch_size 16
```

Train a larger model
```bash
python main.py \
    --model_config peft_pretraining/llama_3b.json \
    --use_peft --train_ln \
    --device cuda:0 \
    --lr 0.0005 \
    --batch_size 16 \
    --gradient_accumulation 4 \
    --num_training_steps 50000
```
