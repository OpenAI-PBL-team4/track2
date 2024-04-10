import time
import torchsummary
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, default_data_collator, TrainingArguments, AutoModel
import peft
import os
import pickle
from itertools import chain
import evaluate


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
# model3 = AutoModel.from_pretrained('./track2/demo/datas/tmp_trainer/checkpoint-5000/')
from datasets import load_dataset
# raw_datasets = load_dataset(".track2/demo/datas/wikitext", cache_dir="./track2/demo/datas/cache_data")
raw_datasets = load_dataset("../../../track2/demo/datas/wikitext", cache_dir="./cache_data")

# indices = torch.randperm(len(raw_datasets['train']))[:3000]
# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = list(raw_datasets["train"].features)
text_column_name = "text"


def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output


tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)


f = open("tokenized_data.pkl", "wb")
pickle.dump(tokenized_datasets, f)
f.close()
f = open("tokenized_data.pkl", "rb")
tokenized_datasets = pickle.load(f)
f.close()
tokenizer.pad_token = tokenizer.eos_token
max_pos_embeddings = 1024
block_size = tokenizer.model_max_length

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {block_size}",
)

f.close()
f = open("grouped_lm_datasets.pkl", "wb")
pickle.dump(lm_datasets, f)
f.close()
f = open("grouped_lm_datasets.pkl", "rb")
lm_datasets = pickle.load(f)
f.close()
train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]
# test_dataset = lm_datasets["test"]




def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


# metric = evaluate.load("./track2/demo/datas/metrics/accuracy")
metric = evaluate.load("./metrics/accuracy")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return metric.compute(predictions=preds, references=labels)

trainer = Trainer(
    model=model2,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)
trainer.args.save_steps= 1000
import math
metrics = trainer.evaluate()
try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")
print(perplexity)
trainer.train()
model2.save_pretrained("./new")
metrics = trainer.evaluate()

try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")
print(perplexity)
model2.save_pretrained("./new")
