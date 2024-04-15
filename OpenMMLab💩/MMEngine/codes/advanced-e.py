import torch.nn as nn
from mmengine.optim import build_optim_wrapper, DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.logging import print_log


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class LayerDecayOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        super().__init__(optim_wrapper_cfg, paramwise_cfg=None)
        self.decay_factor = paramwise_cfg.get('decay_factor', 0.5)

        super().__init__(optim_wrapper_cfg, paramwise_cfg)

    def add_params(self, params, module, prefix='', lr=None):
        if lr is None:
            lr = self.base_lr

        for name, param in module.named_parameters(recurse=False):
            param_group = dict()
            param_group['params'] = [param]
            param_group['lr'] = lr
            params.append(param_group)
            full_name = f'{prefix}.{name}' if prefix else name
            print_log(f'{full_name} : lr={lr}', logger='current')

        for name, module in module.named_children():
            chiled_prefix = f'{prefix}.{name}' if prefix else name
            
            self.add_params(params, module, 
                            chiled_prefix, 
                            lr=lr * self.decay_factor)


class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleDict(dict(linear=nn.Linear(1, 1)))
        self.linear = nn.Linear(1, 1)


model = ToyModel()

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    paramwise_cfg=dict(decay_factor=0.5),
    constructor='LayerDecayOptimWrapperConstructor')

optimizer = build_optim_wrapper(model, optim_wrapper)
