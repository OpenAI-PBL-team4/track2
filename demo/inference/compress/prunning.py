import torch.nn.utils.prune as prune
import transformers
import torch
import torch_pruning

class MyMagnitudeImportance(torch_pruning.importance.Importance):
    pass

def sparsity(name, module):
    return "Sparsity in " + name + ": {:.2f}%".format(
            100. * float(torch.sum(module.weight == 0))
            / float(module.weight.nelement())
        ) + "\n"

def apply_pruning_old(model):
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


def apply_pruning(model, example_inputs, pruning_rate, pruning_iteration):

    pruning_result = ""
    pruning_result += "#################Pruning#####################\n"
    imp = torch_pruning.importance.MagnitudeImportance(p=2)
    ignored_layers = [model.lm_head]

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
            if pruner.DG.check_pruning_group(group):  # 避免将通道剪枝到0
                group.prune()
        macs, nparams = torch_pruning.utils.count_ops_and_params(model, example_inputs)
        pruning_result += "[Loop "+ str(i) +"]macs:"+ str(macs/1024/1024)+ \
                          "mb,nparams:"+ str(nparams/1024/1024)+ "mb\n"

    # DG = torch_pruning.DependencyGraph()
    # DG.build_dependency(model, example_inputs=example_inputs)
    # # group = DG.get_pruning_group(model.transformer.h[0].attn.c_attn,
    # #                              torch_pruning.prune_linear_out_channels,
    # #                              idxs=[2, 6, 9])
    # group = DG.get_pruning_group(model.transformer.wte,
    #                              torch_pruning.prune_embedding_out_channels,
    #                              idxs=[2, 6, 9])
    # for group in DG.get_all_groups(ignored_layers=[model.lm_head], root_module_types=[torch.nn.Embedding, torch.nn.LayerNorm]):
    #     idxs = [2, 4, 6]  # your pruning indices
    #     group.prune(idxs=idxs)
    #     print(group)
    # return group.__str__()

    pruning_result += "#################Pruned#######################\n"

    return pruning_result

def check(model, example_inputs):
    a1 = model.transformer.wte(example_inputs)
    a2 = model.transformer.wpe(a1.long())
    a3 = model.transformer.h[0].ln_1(a2)
    a5 = model.transformer.h[0].attn.c_attn(a3)
    a6 = model.transformer.h[0].attn.c_proj(a5)
