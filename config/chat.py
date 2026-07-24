import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

MODEL_NAME = "SupraLabs/Supra-1.5-50M-Base-exp"
TEXT_FILE = "data/chat/input.txt"
OUTPUT_DIR = "out-discord"
BLOCK_SIZE = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("text", data_files={"train": TEXT_FILE})

def tokenize_fn(examples):
    return tokenizer(examples["text"])

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

lm_dataset = tokenized.map(group_texts, batched=True)

split = lm_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
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
    fp16=torch.cuda.is_available(),
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
