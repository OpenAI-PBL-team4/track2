import transformers
import torch

class M(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.wte = model.transformer.wte
        # self.wpe = model.transformer.wpe
        # self.drop = model.transformer.drop
        self.model = model
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()

        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

def apply_quantiztion(model, input):
    pass
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
