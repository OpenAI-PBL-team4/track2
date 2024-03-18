import time
import torchsummary
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
from track2.demo.inference.compress.prunning import apply_pruning
from track2.demo.inference.compress.quantization import apply_quantiztion

# set mirror for downloading the model
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com/"
# set the max number of loops
STEP_LIMIT = 50

# load tokenizer and model online
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)

# load tokenizer and model offline
print("loading tokenizer and model")
tokenizer = GPT2Tokenizer.from_pretrained('.././gpt2')
model = GPT2LMHeadModel.from_pretrained('.././gpt2', torchscript=True)

print("completed loading")

model.eval()

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

    warm_up(in_tokens)

    with torch.autograd.profiler.profile(enabled=True, use_cuda=use_gpu, record_shapes=False,
                                         profile_memory=False) as prof:

        with torch.no_grad():
            while step < STEP_LIMIT:
                logits, _ = model(in_tokens)
                # choose the highest score result
                out_token = torch.argmax(logits[-1, :], dim=0, keepdim=True)
                in_tokens = torch.cat((in_tokens, out_token), 0)

                # print intermediate text
                # text = tokenizer.decode(in_tokens)
                # print(f"step {step} input: {text}", flush=True)

                step += 1
        out_text = tokenizer.decode(in_tokens)
    # print(prof.table())
    return out_text, prof.table(), summary, compress_result

def compress_model(use_pruning, use_quantization, use_knowledge_distillation):
    # apply 3 different fine-tune technique by modify the model

    # logs when applying the techniques
    compress_result = ""
    global model
    if use_pruning:
        compress_result += apply_pruning(model)
    if use_quantization:
        dimmy_input = torch.tensor(tokenizer.encode("hi"))
        apply_quantiztion(model, dimmy_input)
    if use_knowledge_distillation:
        apply_quantiztion(model)

    return compress_result

def warm_up(in_tokens):
    # Warn-up gpu, cpu
    for _ in range(5):
        start = time.time()
        outputs = model(in_tokens)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end - start) * 1000))






