import time
import torchsummary
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import peft
import os
import pickle

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

lora_config = peft.LoraConfig(
         r=8,
         lora_alpha=32,
         target_modules=["lm_head"],
         lora_dropout=0.05,
         task_type="CAUSAL_LM",
         bias="none"
     )

tokenizer = GPT2Tokenizer.from_pretrained('../../../track2/gpt2')
model = GPT2LMHeadModel.from_pretrained('../../../track2/gpt2', torchscript=True)
model2 = peft.get_peft_model(model, lora_config)

from datasets import load_dataset
raw_datasets = load_dataset("../../../track2/demo/datas/wikitext", cache_dir="./track2/demo/datas/cache_data")

# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]


def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    # clm input could be much much longer than block_size
    return output


tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)
max_pos_embeddings = 1024

f = open("tokenized_data2.pkl", "wb")
pickle.dump(tokenized_datasets, f)
f.close()
# f = open("tokenized_data.pkl", "rb")
# a = pickle.load(tokenized_datasets, f)
# f.close()