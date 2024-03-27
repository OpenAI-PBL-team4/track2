config = GPT2Config.from_pretrained(
    "gpt2",
    lora=True,
    lora_rank=4,
    lora_alpha=32,
)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
