# Efficient and Effective Model Compression
OpenAI PBL track2 project


# Script 1:webui.py  

set up a webui for collect input text and compression config by user,  
call and give inputs to "inference.py" script.  
show the output inference result, model size, inference time from "inference.py"  

use command "python webui.py" to run the script.  
and then go to 127.0.0.1:7860 or localhost:7860 to use the webui.  

output a webui which user can input text, select compression config  
and show the results from inference.py.  

  
	  
# Script 2: inference.py  

load the Causal LM, tokenizer, apply lora, warmup CPU and CUDA.  
use compression configs to decide whether call other scripts.  
handle the inference of the Causal LM.  

return output back to webui.   
will be called automatically by webui.py  
	 
result text after inference  
perplexity of the LM model calculated by trainer  
CPU, CUDA operators total time, number of calls  
model size, change of model size after each compression stage  
  
  

  

# Script 3: conv1d_to_linear.py  

convert conv1d layers in GPT2 to Linear layer and transpose the weight.  
It will be easier to do pruning and quantization on Linear instead of conv1d.  

will be called automatically by inference.py if use one of use_pruning, checked use_quantization is checked by user.  

return a model that converted all conv1d into linear layer.  


   


# Script 4: pruning.py

use torch-pruning by VainF to apply a structural pruning by creating a Dependency Graph of the model.  
user can choose the proportion of the model(1-99%) and iterations(1-30) to prune.  

will be called automatically by inference.py if use checked use_pruning  

return a model after pytorch dynamic ptq to inference.  

  


# Script 5: quantization.py

use pytorch dynamic ptq to quantize the model.  
user can choose to quantize to int8 or fp16  

will be called automatically by inference.py if use checked use_quantization  

return a model after pytorch dynamic ptq to inference.  

  

# Script 6: lora_training.py

an extra script that used to train a lora.  
it will use peft to inject a rank-8 lora for the LM head of the model.  
and train the lora with wikitext datasets, then save it locally.  

use command "python lora_training.py" to run the script, wait until it finished training.

output an adapter_model.safetensor and adapter_config.json for the lora.

   

# Script 7: get_trainer.py  

include functions to create a trainer with LLM for training or evaluation, and calculate perplexity.  

will be called by inference for evaluation when "evaluate_perplexity" is checked  
or called by "lora_training.py" for training of a lora.  

it will output a trainer for training or evaluation.
  


# Script 8: preprocess_data.py

load and tokenize wikitext datasets, and concatenate them into chunks with size of 1024.

will be called by get_trainer.py

it will output a tokenized and concatenated wikitext dataset with block size 1024.
  
  


# members:  
qyh080821 and LTC = Tiancheng Li  
Sgz26013 = Guozhi Su  
Chelseo = Leyi Yu  
SlingmaoS = Jinfeng Xu
