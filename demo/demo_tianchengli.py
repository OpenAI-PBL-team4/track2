import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gradio as gr
STEP_LIMIT = 50
tokenizer = GPT2Tokenizer.from_pretrained('.././gpt2')

model = GPT2LMHeadModel.from_pretrained('.././gpt2', torchscript=True)
default_text= "Nya haha! You may write something here. But don't expect GPT2 to give you some excellent results. :("
model.eval()

input_custom = gr.Textbox(placeholder=default_text)

def greet(text):
    in_tokens = torch.tensor(tokenizer.encode(text))
    out_token = 0
    step = 0
    with torch.no_grad():
        while step < STEP_LIMIT:
            logits, _ = model(in_tokens)
            out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
            in_tokens = torch.cat((in_tokens, out_token), 0)
            text = tokenizer.decode(in_tokens)
            print(f"step {step} input: {text}", flush=True)
            step += 1
    out_text = tokenizer.decode(in_tokens)
    return out_text

demo = gr.Interface(fn=greet, inputs=input_custom, outputs="textbox")
demo.launch(debug=True)

