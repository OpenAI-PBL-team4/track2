import torch.nn.utils.prune as prune
import transformers
import torch
import torch_pruning


def sparsity(name, module):
    return "Sparsity in " + name + ": {:.2f}%".format(
            100. * float(torch.sum(module.weight == 0))
            / float(module.weight.nelement())
        ) + "\n"


def apply_pruning(model, example_inputs, pruning_rate, pruning_iteration):

    pruning_result = ""
    pruning_result += "#################Pruning#####################\n"
    imp = torch_pruning.importance.MagnitudeImportance(p=2)
    ignored_layers = [model.lm_head, model.transformer.wte, model.transformer.wpe]

    for m in model.modules():
        if isinstance(m, transformers.models.gpt2.modeling_gpt2.GPT2Attention):
            ignored_layers.append(m)
    iterative_steps = pruning_iteration
    pruner = torch_pruning.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_rate,
        ignored_layers=ignored_layers,
    )

    # base_macs, base_nparams = torch_pruning.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):

        for group in pruner.step(
                interactive=True):
            print(group.details())
            if pruner.DG.check_pruning_group(group):
                group.prune()
        macs, nparams = torch_pruning.utils.count_ops_and_params(model, example_inputs)
        pruning_result += "[Loop "+ str(i) +"]macs:"+ str(macs/1024/1024)+ \
                          "mb,nparams:"+ str(nparams/1024/1024)+ "mb\n"

    pruning_result += "#################Pruned#######################\n"

    return pruning_result

def check(model, example_inputs):
    a1 = model.transformer.wte(example_inputs)
    a2 = model.transformer.wpe(a1.long())
    a3 = model.transformer.h[0].ln_1(a2)
    a5 = model.transformer.h[0].attn.c_attn(a3)
    a6 = model.transformer.h[0].attn.c_proj(a5)
