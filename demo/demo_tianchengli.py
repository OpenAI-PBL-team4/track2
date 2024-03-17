import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gradio as gr
import os
import time
import sys
import torchsummary
import torch.nn.utils.prune as prune

# set mirror for downloading the model
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com/"
# set the max number of loops
STEP_LIMIT = 50

# load tokenizer and model

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)
print("loading tokenizer and model")
tokenizer = GPT2Tokenizer.from_pretrained('.././gpt2')
model = GPT2LMHeadModel.from_pretrained('.././gpt2', torchscript=True)

print("completed loading")

default_text = "Nya haha! You may write something here. But don't expect GPT2 to give you some excellent results. :("

model.eval()

input_custom = gr.Textbox(placeholder=default_text)
output_result = gr.Textbox(label="result")
output_time = gr.Textbox(label="time analysis", scale=2)
output_size = gr.Textbox(label="size analysis")
output_compression = gr.Textbox(label="compression analysis")

def greet(text, use_pruning, use_quantization, use_knowledge_distillation, use_gpu):
    device = torch.device('cpu')
    compress_result = compress_model(use_pruning, use_quantization, use_knowledge_distillation)

    if use_gpu:
        device = torch.device('cuda')
    in_tokens = torch.tensor(tokenizer.encode(text)).to(device)
    out_token = 0
    step = 0
    summary = ""
    if use_gpu:
        summary = torchsummary.summary(model, in_tokens)
    # Warn-up
    for _ in range(5):
        start = time.time()
        outputs = model(in_tokens)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end - start) * 1000))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=use_gpu, record_shapes=False,
                                         profile_memory=False) as prof:

        with torch.no_grad():
            while step < STEP_LIMIT:
                logits, _ = model(in_tokens)
                # choose the highest score result
                out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
                in_tokens = torch.cat((in_tokens, out_token), 0)
                text = tokenizer.decode(in_tokens)
                # print(f"step {step} input: {text}", flush=True)
                step += 1
        out_text = tokenizer.decode(in_tokens)
    # print(prof.table())
    return out_text, prof.table(), summary, compress_result

def compress_model(use_pruning, use_quantization, use_knowledge_distillation):
    compress_result = ""
    if use_pruning:
        compress_result += apply_pruning(model)
    if use_quantization:
        apply_quantiztion(model)
    if use_knowledge_distillation:
        apply_quantiztion(model)
    return compress_result

def sparsity(name, module):
    return "Sparsity in " + name + ": {:.2f}%".format(
            100. * float(torch.sum(module.weight == 0))
            / float(module.weight.nelement())
        ) + "\n"

def apply_quantiztion(model):
    pass

def apply_pruning(model):
    prunning_result = ""
    print("#################Pruning#####################\n")
    prunning_result += "#################Pruning#####################\n"
    for name, module in model.named_modules():
        # prune 20% of connections in all conv1d layers, 40% in linear layers
        # try:
        #     prune.l1_unstructured(module, name='weight', amount=0.2)
        #     print("A pruned", name, type(module))
        # except:
        #     pass

        if isinstance(module, transformers.pytorch_utils.Conv1D):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prunning_result += sparsity(name, module)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.4)
            prunning_result += sparsity(name, module)
    prunning_result+="#################Pruned#######################\n"



    print("#################Pruned#######################\n")
    return prunning_result


def apply_knowledge_distillation(model):
    pass


demo = gr.Interface(fn=greet, inputs=[input_custom, "checkbox", "checkbox", "checkbox","checkbox"],
                    outputs=[output_result, output_time, output_size, output_compression])
demo.launch(debug=True)


