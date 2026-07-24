"""
Prepare the Discord chat dataset for fine-tuning GPT-2 with nanoGPT.
Encodes input.txt using the GPT-2 BPE tokenizer (tiktoken) so it's
compatible with pretrained GPT-2 weights (init_from='gpt2').
Saves train.bin and val.bin containing the token ids.
"""
import os
import requests
import numpy as np
import tiktoken

# Download or locate the discord chat export dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/play-station451/Hdjdjdj/refs/heads/main/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# Encode with the GPT-2 BPE tokenizer (matches pretrained GPT-2 weights)
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# No meta.pkl is saved here on purpose: without it, nanoGPT's sample.py
# falls back to GPT-2's own tokenizer for decoding, which is what we want
# since we're fine-tuning pretrained GPT-2 rather than training from scratch.
