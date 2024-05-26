import torch
from safetensors.torch import save_file

state_dict = torch.load("step=00144000.ckpt", map_location='cpu')
save_file(state_dict["state_dict"], "diffusion_pytorch_model.safetensors", None)