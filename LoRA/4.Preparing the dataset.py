class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc=f"Loading {os.path.basename(file_path)}"):
                tokenized_text = tokenizer.encode(line, add_special_tokens=True, truncation=True, max_length=block_size)
                self.examples.append(tokenized_text)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return {'input_ids': torch.tensor(self.examples[i], dtype=torch.long)}

train_file_path = 'D:\\wet_files\\train_part_0.txt'#改成自己的路径
valid_file_path = 'D:\\wet_files\\valid_part_0.txt'#同上

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file_path,
    block_size=128
)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=valid_file_path,
    block_size=128
)
