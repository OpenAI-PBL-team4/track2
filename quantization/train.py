import torch.nn.functional as F
import copy
import torchvision
from torchvision import transforms
from torch.quantization import get_default_qconfig
from torch import optim
import os
import time
import transformers
import platform

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

def choose_backend():
  machine = platform.machine()
  is_x86 = machine.startswith('x86') or machine.startswith('i386') or machine.startswith('i686')
  is_arm = machine.startswith('arm') or machine.startswith('aarch')

  if is_x86:
      backend = 'fbgemm'
      print("system with x86")
  elif is_arm:
      backend = 'qnnpack'
      print("system with ARM")
  else:
      print("system is not x86 or ARM")
      backend = None
  return backend

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

def print_use_time(model,input_id,label=""):
  start_time = time.time()
  with torch.no_grad():
    original_output = model.generate(input_ids)
    time = time.time() - start_time
  return time

if __name__ == "main":

  ## PTQ
  basic_information()
  backend = choose_backend()
  qconfig = torch.quantization.get_default_qconfig(backend)

  model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
  tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
  
  input_text = "I am going to fly,"
  input_id = tokenizer.encode(input_text, return_tensors="pt")
  orig_use_time = print_use_time(model,input_id)
  orig_memory = print_size_of_model(model,"original")

  quantized_model = dynamic_quantization(model, torch.nn.Linear, dtype=torch.qint8)
  quantized_use_time = print_use_time(quantized_model,input_id)
  quan_memory = print_size_of_model(quantized_model,"int8")

  reduced_memory = orig_memory/quan_memory
  reduced_time = orig_use_time - quantized_use_time
  
  print(f"Time changes from {orig_use_time:.2f} to {quantized_use_time:.2f}")
  print(f" This reduces {reduced_time:.2f} seconds")}
  
  total_parameters = model.num_parameters()
  print("Total parameters in 8b:", total_parameters)
