import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.deepspeed import HfDeepSpeedConfig

# DeepSpeed config path (external JSON file recommended)
ds_config_file = "deepspeed_config.json"
hf_deepspeed_config = HfDeepSpeedConfig(ds_config_file)

# Load tokenizer and model (ensure weights are available)
model_name_or_path = "path/to/llama-72b"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto",
    use_cache=False
)

# Load Abheja dataset (assumes a proper HF dataset or JSON format)
data = load_dataset("abheja_dataset_script_or_json")

# Preprocessing
max_length = 2048
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

tokenized_data = data.map(tokenize, batched=True, remove_columns=data["train"].column_names)

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output/llama72b-abheja",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    evaluation_strategy="no",
    deepspeed=ds_config_file,
    fp16=True,
    report_to="none",
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    resume_from_checkpoint=get_last_checkpoint("./output/llama72b-abheja") if os.path.exists("./output/llama72b-abheja") else None,
)

# Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_data["train"],
    data_collator=collator,
)

# Launch training
trainer.train()

# Save model
trainer.save_model("./final-llama72b-abheja")
tokenizer.save_pretrained("./final-llama72b-abheja")
