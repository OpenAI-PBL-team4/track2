import torch.nn.utils.prune as prune
import transformers
import torch

def sparsity(name, module):
    return "Sparsity in " + name + ": {:.2f}%".format(
            100. * float(torch.sum(module.weight == 0))
            / float(module.weight.nelement())
        ) + "\n"

def apply_pruning(model):
    prunning_result = ""
    print("#################Pruning#####################\n")
    prunning_result += "#################Pruning#####################\n"
    for name, module in model.named_modules():
        # prune 20% of connections in all conv1d layers
        # try:
        #     prune.l1_unstructured(module, name='weight', amount=0.2)
        #     print("A pruned", name, type(module))
        # except:
        #     pass

        if isinstance(module, transformers.pytorch_utils.Conv1D):
            prune.l1_unstructured(module, name='weight', amount=0.2)
            prunning_result += sparsity(name, module)

        # elif isinstance(module, torch.nn.Linear):
        #     prune.l1_unstructured(module, name='weight', amount=0.4)
        #     prunning_result += sparsity(name, module)

    prunning_result += "#################Pruned#######################\n"
    print("#################Pruned#######################\n")

    return prunning_result