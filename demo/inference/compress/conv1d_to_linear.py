import transformers
import torch

def _conv1d_to_linear(module):
    in_size, out_size = module.weight.shape
    linear = torch.nn.Linear(in_size, out_size)
    linear.weight.data = module.weight.data.T.contiguous()
    linear.bias.data = module.bias.data
    return linear

def conv1d_to_linear(model):
    """find all the conv1d and tranform into nn.linear and transpose the weight
    """
    for name in list(model._modules):
        if name == "lm_head":
            continue
        module = model._modules[name]
        if isinstance(module, transformers.modeling_utils.Conv1D):
            linear = _conv1d_to_linear(module)
            model._modules[name] = linear
        else:
            conv1d_to_linear(module)