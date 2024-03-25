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
  quantized_model = torch.quantization.quantize_dynamic(model, {layer}, dtype=mode)
  return quantized_model

def basic_information():
  print("transformers version:", transformers.__version__)
  print("torch version:", torch.__version__)
  if torch.cuda.is_available():
      num_devices = torch.cuda.device_count()
      print("Available Gpus", num_devices)
      for i in range(num_devices):
          device = torch.cuda.get_device_properties(i)
          print(f"\nGPU {i} detail:")
          print("name:", device.name)
          print("capable:", f"{device.major}.{device.minor}")
          print("GB:", round(device.total_memory / (1024**3), 1))
      else:
          print("no use GPU")    

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

def print_use_time(model,input_text):
  start_time = time.time()
  input_ids = tokenizer.encode(input_text, return_tensors="pt")
  with torch.no_grad():
    output = model.generate(input_ids)
  use_time = time.time() - start_time
  return use_time

if __name__ == "main":

  basic_information()
  
  model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
  tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

  
  
  total_parameters = model.num_parameters()
  print("Total parameters in 8b:", total_parameters)
