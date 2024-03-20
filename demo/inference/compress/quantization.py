import torch
import copy

type_table = {#"torch.quint8":torch.quint8,
              "torch.qint8":torch.qint8,
              #"torch.qint32":torch.qint32,
              "torch.float16":torch.float16}


def apply_quantiztion(model, input, quant_dtype):

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
