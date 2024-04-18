import torch
import copy
import torch.nn as nn
import torch.ao.quantization as quantization
from transformers import GPT2LMHeadModel, GPT2Config
import torch.nn.functional as F
from torchvision import transforms
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fx.graph_module import ObservedGraphModule
from torch.quantization import get_default_qconfig
from torch import optim
import time
import transformers
import conv1d_to_linear
type_table = {#"torch.quint8":torch.quint8,
              "torch.qint8":torch.qint8,
              #"torch.qint32":torch.qint32,
              "torch.float16":torch.float16}


def apply_quantiztion(model, type,input, quant_dtype,mode = None):
    if type == "dynamic":
	    quantization_result = ""
	    quantization_result += "#################Quantizing#####################\n"
	    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=type_table[quant_dtype])
	    model._modules['transformer'] = quantized_model._modules['transformer']
	    quantization_result += "#################Quantized#####################\n"
	    return quantization_result
	    # model_origin = M(model)
	    # model_origin.eval()
	    # model_origin.qconfig = torch.quantization.get_default_qconfig("qnnpack")
	    # # model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
	    # model_origin.model.transformer.wte.qconfig = None
	    # model_origin.model.transformer.wpe.qconfig = None
	    # model_prepared = torch.ao.quantization.prepare(model_origin)
	    #
	    # model_int8 = torch.ao.quantization.convert(model_prepared)
	    # return model_int8
elif type == "static":
	if mode == "eager":
		eager_quant(model)
	else:
		fx_quant(model)
else:
	print("Wrong type")


class QuantizedGPT2(nn.Module):
    def __init__(self, model):
        super(QuantizedGPT2, self).__init__()
        self.quant = quantization.QuantStub()
        self.model = model
        self.dequant = quantization.DeQuantStub()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        input_ids = self.quant(input_ids)
        print(input_ids)
        attention_mask = self.quant(attention_mask) if attention_mask is not None else None
        outputs = self.model.forward(input_ids)
        outputs = self.dequant(outputs[0])
        return outputs

def qconfig_set(backend = "fbgemm",**parameters):

	if backend == 'fbgemm':
		qconfig = torch.quantization.get_default_qconfig('fbgemm')
		
	elif backend == 'qnnpack':
		qconfig = torch.quantization.get_default_qconfig('qnnpack')
		
	elif backend == 'onednn':
		qconfig = torch.quantization.get_default_qconfig('onednn')
		
	elif backend == "others":
		qconfig = Qconfig(weight = usr_weight,
						 activation = usr_activation)
    return qconfig


def calibration(model_pre):
	global model	
	
	raw_datasets = load_dataset("./datas/wikitext")
    validation_data, train_data = preprocess(raw_datasets, tokenizer)
	
	for i in range(0,400):
		if torch.numel(train_data.iloc[i]["input_ids"]) !=0:
			outputs = model_pre(tokenized_data.iloc[i]["input_ids"])

def eager_quant(model):

	quant_model = QuantizedGPT2(model)
	
	quant_model.qconfig = qconfig_set(backend,parameters)
	model_pre = torch.quantization.prepare(quant_model, inplace=False)
	
	model_pre = calibration(model_pre)
	model_quantized  = torch.quantization.convert(model_pre, inplace=False,remove_qconfig=True) 
	
def fx_quant(model):
	 qconfig_dict = {
        "":qconfig_set(backend,parameters),
        "object_type": [
            (model.transformer.h[0].ln_1, None),   #modules to jump
			(model.transformer.h[1].ln_1, None), 
			(model.transformer.h[2].ln_1, None), 
			(model.transformer.h[3].ln_1, None), 
			(model.transformer.h[4].ln_1, None), 
			(model.transformer.h[5].ln_1, None), 
			(model.transformer.h[6].ln_1, None), 
			(model.transformer.h[7].ln_1, None), 
			(model.transformer.h[8].ln_1, None), 
			(model.transformer.h[9].ln_1, None), 
			(model.transformer.h[10].ln_1, None),
			(model.transformer.h[11].ln_1, None)
        ],
    }
	text = "One example needs to expand,"
	example_in = tokenizer(text, return_tensors='pt').input_ids
	model_pre = prepare_fx(model_to_quantize, qconfig_dict,example_in)
	
	model_pre = calibration(model_pre)
	
    model_quantized  = torch.quantization.convert_fx(model_pre, inplace=False,remove_qconfig=True) 
