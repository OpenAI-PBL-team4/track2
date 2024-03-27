trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data])}
)

trainer.train()
