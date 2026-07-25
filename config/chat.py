import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

MODEL_NAME = "SupraLabs/MicroSupra-1k"
TEXT_FILES = [
    "https://raw.githubusercontent.com/play-station451/Hdjdjdj/refs/heads/main/input.txt",
    "https://raw.githubusercontent.com/play-station451/Hdjdjdj/refs/heads/main/input2.txt",
    "https://raw.githubusercontent.com/play-station451/Hdjdjdj/refs/heads/main/input3.txt"
]
OUTPUT_DIR = "out-supra-chat"
BLOCK_SIZE = 512

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

model.config.pad_token_id = tokenizer.pad_token_id

dataset = load_dataset("text", data_files={"train": TEXT_FILES})

def tokenize_fn(examples):
    texts_with_eos = [t + tokenizer.eos_token for t in examples["text"] if t.strip()]
    return tokenizer(texts_with_eos)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

def group_texts(examples):
    concatenated = sum(examples["input_ids"], [])
    total_len = (len(concatenated) // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        "input_ids": [
            concatenated[i : i + BLOCK_SIZE] for i in range(0, total_len, BLOCK_SIZE)
        ]
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized.map(
    group_texts,
    batched=True,
    remove_columns=tokenized["train"].column_names,
)

split = lm_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    logging_steps=10,
    learning_rate=3e-5,
    warmup_steps=50,
    max_grad_norm=1.0,
    fp16=use_fp16,
    bf16=use_bf16,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Model saved to {OUTPUT_DIR}")
