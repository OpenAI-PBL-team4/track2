
import evaluate
from itertools import chain
import pickle
def preprocess(raw_datasets, tokenizer):
    def tokenize_function(examples):
        output = tokenizer(examples["text"])
        # clm input could be much much longer than block_size
        return output


    try:
        f = open("tokenized_data2.pkl", "rb")
        tokenized_datasets = pickle.load(f)
        f.close()
    except:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
        f = open("tokenized_data.pkl", "wb")
        pickle.dump(tokenized_datasets, f)
        f.close()

    block_size = 1024
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # total_length = (total_length // block_size) * block_size
        total_length -= (total_length % block_size)
        # Split by chunks of 1024.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    try:
        f = open("grouped_lm_datasets.pkl", "rb")
        lm_datasets = pickle.load(f)
        f.close()
    except:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )
        f = open("grouped_lm_datasets.pkl", "wb")
        pickle.dump(lm_datasets, f)
        f.close()


    return lm_datasets["validation"], lm_datasets["test"]