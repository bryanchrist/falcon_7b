import os
import sys
import builtins
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
#from adapter-transformers import AdapterType, AdapterConfig, load_adapter

# Set the environment variable
os.environ["HF_REMOTES_OFFLINE"] = "1"

# Redirect stdin to /dev/null
sys.stdin = open(os.devnull)

model_path = "checkpoints/tiiuae/falcon-7b"  # Specify the path to the downloaded model
adapter_path = "output/checkpoint-3250"  # Specify the path to the adapter weights
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Patch the built-in input function to return 'y' automatically
def mock_input(prompt=None):
    return 'y'

# Patch the input function to use the mock_input function
builtins.input = mock_input

try:
    # Attempt to load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
       # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    )
    model.to('cuda')
except EOFError:
    # If an EOFError occurs, provide the expected input ('y')
    pass

# Restore stdin
sys.stdin = sys.__stdin__

# Load the adapter weights
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

for i in range(0, 10):
    prompt = "Write a grade 4 Multiplication question and corresponding equation to solve the problem."
    formatted_prompt = (f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: ")
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    inputs = inputs.to('cuda')
    attention_mask = attention_mask.to('cuda')
    output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 100)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(generated_text)
    output_file = "output.txt"  # Specify the path and filename for the output file
    with open(output_file, "a") as f:
        f.write(generated_text)

print("Generated text saved to", output_file)


adapter_path = "output/checkpoint-250"  # Specify the path to the adapter weights
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Patch the built-in input function to return 'y' automatically
def mock_input(prompt=None):
    return 'y'

# Patch the input function to use the mock_input function
builtins.input = mock_input

try:
    # Attempt to load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
       # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    )
    model.to('cuda')
except EOFError:
    # If an EOFError occurs, provide the expected input ('y')
    pass

# Restore stdin
sys.stdin = sys.__stdin__

# Load the adapter weights
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

for i in range(0, 10):
    prompt = "Write a grade 4 Multiplication question and corresponding equation to solve the problem."
    formatted_prompt = (f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: ")
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    inputs = inputs.to('cuda')
    attention_mask = attention_mask.to('cuda')
    output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 100)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(generated_text)
    output_file = "output.txt"  # Specify the path and filename for the output file
    with open(output_file, "a") as f:
        f.write(generated_text)

print("Generated text saved to", output_file)

adapter_path = "output/checkpoint-9000"  # Specify the path to the adapter weights
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Patch the built-in input function to return 'y' automatically
def mock_input(prompt=None):
    return 'y'

# Patch the input function to use the mock_input function
builtins.input = mock_input

try:
    # Attempt to load the model with trust_remote_code=True
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
       # max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        config=AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    )
    model.to('cuda')
except EOFError:
    # If an EOFError occurs, provide the expected input ('y')
    pass

# Restore stdin
sys.stdin = sys.__stdin__

# Load the adapter weights
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

for i in range(0, 10):
    prompt = "Write a grade 4 Multiplication question and corresponding equation to solve the problem."
    formatted_prompt = (f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n### Response: ")
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    inputs = inputs.to('cuda')
    attention_mask = attention_mask.to('cuda')
    output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 100)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(generated_text)
    output_file = "output.txt"  # Specify the path and filename for the output file
    with open(output_file, "a") as f:
        f.write(generated_text)

print("Generated text saved to", output_file)