import time
import torchsummary
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
from track2.demo.inference.compress.prunning import apply_pruning
from track2.demo.inference.compress.quantization import apply_quantiztion
from track2.demo.inference.compress.knowledge_distillation import apply_knowledge_distillation
from track2.demo.inference.compress.conv1d_to_linear import conv1d_to_linear
# set mirror for downloading the model
os.environ["HF_ENDPOINT"] = "http://hf-mirror.com/"
# set the max number of loops
STEP_LIMIT = 30

# load tokenizer and model online
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)
model = None
tokenizer = GPT2Tokenizer.from_pretrained('.././gpt2')


def greet(text, use_pruning, pruning_rate, pruning_iteration,
          use_quantization, quant_dtype,
          use_knowledge_distillation,
          use_gpu):

    device = torch.device('cpu')
    if use_gpu:
        device = torch.device('cuda')
    # load tokenizer and model offline
    print("loading model")
    load_model()
    print("completed loading")

    # fine-tune
    compress_result = compress_model(use_pruning, pruning_rate, pruning_iteration,
                                     use_quantization, quant_dtype,
                                     use_knowledge_distillation,
                                     use_gpu)



    global model
    model = model.to(device)
    in_tokens = torch.tensor(tokenizer.encode(text)).to(device)
    out_token = 0
    step = 0
    summary = ""
    summary = size_check(model)

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

def load_model():
    global model
    # load tokenizer and model offline
    print("loading tokenizer and model")
    tokenizer = GPT2Tokenizer.from_pretrained('.././gpt2')
    model = GPT2LMHeadModel.from_pretrained('.././gpt2', torchscript=True)

    print("completed loading")

    model.eval()


def compress_model(use_pruning, pruning_rate, pruning_iteration,
                   use_quantization, quant_dtype,
                   use_knowledge_distillation,
                   use_gpu):
    # apply 3 different fine-tune technique by modify the model

    # logs when applying the techniques
    compress_result = ""
    global model
    example_inputs = torch.tensor(tokenizer.encode("hi!")).to(torch.device("cpu") if not use_gpu
                                                              else torch.device("cuda"))
    size_1 = size_check(model)
    size_2 = size_check(model)
    # transpose conv1d to nn.linear before apply finetune
    if use_pruning or use_quantization:
        compress_result += "############Converting conv1d to linear##############\n"
        conv1d_to_linear(model)
        compress_result += show_size_change(model, size_1)
        size_1 = size_check(model)


    if use_pruning:
        compress_result += apply_pruning(model, example_inputs, pruning_rate, pruning_iteration)
        compress_result += show_size_change(model, size_1)
        size_1 = size_check(model)

    if use_quantization:
        if use_gpu:
            compress_result += "Cannot use gpu for pytorch quantization\n"
        else:
            dimmy_input = torch.tensor(tokenizer.encode("hi"))
            compress_result += apply_quantiztion(model, dimmy_input, quant_dtype)
            compress_result += show_size_change(model, size_1)
            size_check(model)

    if use_knowledge_distillation:
        apply_knowledge_distillation(model)
        compress_result += show_size_change(model, size_1)
        size_1 = size_check(model)

    return compress_result

def warm_up(in_tokens):
    # Warn-up gpu, cpu
    for _ in range(5):
        start = time.time()
        model(in_tokens)
        torch.cuda.synchronize()
        end = time.time()
        print('Time:{}ms'.format((end - start) * 1000))

def size_check(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return 'model size: {:.3f}MB'.format(size_all_mb)

def show_size_change(model, size_1):
    size_2 = size_check(model)
    compress_result = size_1 + "->>>>>" + size_2 + "\n"

    return compress_result





