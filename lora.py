from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
dataset_path = "data/ASDiv_clean_formatted.json"

# Load the dataset from a local file or directory
dataset = load_dataset("json", data_files=dataset_path)

#Load the model
model_name='tiiuae/falcon-7b-instruct'

original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Prepare the data 
def tokenize_function(example):
    start_prompt = (f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n'")
    end_prompt = (f"\n\n### Response: ")
    prompt = [start_prompt + instruction + end_prompt for instruction in example["instruction"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["output"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example

# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['instruction', 'input', 'output',])

#set up LoRA and train model
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.AUTO_CAUSAL_LM 
)

peft_model = get_peft_model(original_model, 
                            lora_config)

output_dir = f'./peft-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=4,
    logging_steps=1,
    max_steps=1    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

peft_trainer.train()

peft_model_path="./peft-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

#Load Peft model and generate sample text
from peft import PeftModel, PeftConfig

peft_model_base = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

peft_model = PeftModel.from_pretrained(peft_model_base, 
                                       './peft-checkpoint-local/', 
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)
pipeline = transformers.pipeline(
    "text-generation",
    model=peft_model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
for i in range(0,10):
    prompt = "Write a grade 4 Multiplication question and corresponding equation to solve the problem."
    formatted_prompt = (f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: ")
    sequences = pipeline(
       formatted_prompt,
        max_new_tokens = 100,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        output_file = "output.txt"  # Specify the path and filename for the output file
        with open(output_file, "a") as f:  # Open the file in append mode ("a")
            f.write(f"Result: {seq['generated_text']}" + "\n")  # Append the generated text to the file
        print(f"Result: {seq['generated_text']}")