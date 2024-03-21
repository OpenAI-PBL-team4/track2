import torch.nn.functional as F
import copy
import torchvision
from torchvision import transforms
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fx.graph_module import ObservedGraphModule
from torch.quantization import get_default_qconfig
from torch import optim
import os
import time
import transformers

def dynamic_quantization(model,layer,mode):


def model_quantization_test(model,label):
  for name, param in model.named_parameters():
      if param.dtype == label:
          print(f"Parameter name: {name}")
          print(f"Parameter dtype: {param.dtype}")
          print("Parameter is quantized successfully.")

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

model = transformers.GPT2LMHeadModel.from_pretrained("./autodl-tmp/1_gpt2")
quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

total_parameters = model.num_parameters()
print("Total parameters in 8b:", total_parameters)
