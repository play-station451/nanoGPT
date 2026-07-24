# config for fine-tuning GPT-2 on Discord chat data with nanoGPT
# usage: python train.py config/train_discord.py

import time

out_dir = "out-discord"
eval_interval = 50
eval_iters = 40
wandb_log = False
dataset = "discord"  # expects data/discord/train.bin and data/discord/val.bin

init_from = "gpt2"  # load pretrained GPT-2 weights instead of training from scratch
# use 'gpt2-medium' or 'gpt2-large' instead if you have the GPU memory for a stronger base

# fine-tuning needs a much smaller LR than training from scratch (~6e-4)
learning_rate = 3e-5
decay_lr = False
warmup_iters = 50

# with 2.5MB of text (~600k-700k GPT-2 tokens), don't overtrain --
# start here and watch val loss, stop early if it stops improving
max_iters = 1000
lr_decay_iters = max_iters

batch_size = 4
gradient_accumulation_steps = 8  # effective batch size ~32, adjust down if you run out of GPU memory

always_save_checkpoint = True

# log a bit more often since this is a short run
log_interval = 10
