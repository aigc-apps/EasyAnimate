import torch
from safetensors.torch import load_file, safe_open, save_file

original_safetensor_path = 'diffusion_pytorch_model.safetensors'
new_safetensor_path = 'easyanimate_v1_mm.safetensors'  #

original_weights = load_file(original_safetensor_path)
temporal_weights = {}
for name, weight in original_weights.items():
    if 'temporal' in name:
        temporal_weights[name] = weight
save_file(temporal_weights, new_safetensor_path, None)
print(f'Saved weights containing "temporal" to {new_safetensor_path}')