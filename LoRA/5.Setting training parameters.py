training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=1000,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=500
)
