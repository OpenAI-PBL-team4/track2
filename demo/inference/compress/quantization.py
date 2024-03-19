import transformers
import torch
import copy

type_table = {#"torch.quint8":torch.quint8,
              "torch.qint8":torch.qint8,
              #"torch.qint32":torch.qint32,
              "torch.float16":torch.float16}
def _conv1d_to_linear(module):
    in_size, out_size = module.weight.shape
    linear = torch.nn.Linear(in_size, out_size)
    linear.weight.data = module.weight.data.T.contiguous()
    linear.bias.data = module.bias.data
    return linear

def conv1d_to_linear(model):

    for name in list(model._modules):
        if name == "lm_head":
            continue
        module = model._modules[name]
        if isinstance(module, transformers.modeling_utils.Conv1D):
            linear = _conv1d_to_linear(module)
            model._modules[name] = linear
        else:
            conv1d_to_linear(module)

def apply_quantiztion(model, input, quant_dtype):
    #model2 = copy.deepcopy(model)
    conv1d_to_linear(model)
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=type_table[quant_dtype])
    model._modules['transformer'] = quantized_model._modules['transformer']
    return quantized_model
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
