"""Modified from https://github.com/kijai/ComfyUI-MochiWrapper
"""
import torch
import torch.nn as nn

def autocast_model_forward(cls, origin_dtype, *inputs, **kwargs):
    weight_dtype = cls.weight.dtype
    cls.to(origin_dtype)

    # Convert all inputs to the original dtype
    inputs = [input.to(origin_dtype) for input in inputs]
    out = cls.original_forward(*inputs, **kwargs)

    cls.to(weight_dtype)
    return out

def convert_model_weight_to_float8(model, exclude_module_name='embed_tokens'):
    for name, module in model.named_modules():
        if exclude_module_name not in name:
            for param_name, param in module.named_parameters():
                if exclude_module_name not in param_name:
                    param.data = param.data.to(torch.float8_e4m3fn)

def convert_weight_dtype_wrapper(module, origin_dtype):
    for name, module in module.named_modules():
        if name == "" or "embed_tokens" in name:
            continue
        original_forward = module.forward
        if hasattr(module, "weight"):
            setattr(module, "original_forward", original_forward)
            setattr(
                module,
                "forward",
                lambda *inputs, m=module, **kwargs: autocast_model_forward(m, origin_dtype, *inputs, **kwargs)
            )
