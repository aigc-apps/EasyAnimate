import time
import torch
import gc
import os

from easyanimate.api.api import (infer_forward_api,
                                 update_diffusion_transformer_api,
                                 update_edition_api)
from easyanimate.ui.ui import ui, ui_eas, ui_modelscope

# Set PyTorch to use expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

def check_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def move_to_cpu_if_needed(model):
    if torch.cuda.is_available():
        if torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.9:
            print("GPU memory usage high, moving model to CPU")
            model = model.cpu()
            clear_gpu_memory()
    return model

if __name__ == "__main__":
    # Choose the ui mode  
    ui_mode = "normal"
    
    # GPU memory mode
    GPU_memory_mode = "sequential_cpu_offload"
    
    # Use torch.float16 if GPU does not support torch.bfloat16
    # Some graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16

    # Server ip
    server_name = "0.0.0.0"
    server_port = 7861

    # Params below is used when ui_mode = "modelscope"
    edition = "v5"
    # Config
    config_path = "config/easyanimate_video_v5_magvit_multi_text_encoder.yaml"
    # Model path of the pretrained model
    model_name = "models/Diffusion_Transformer/EasyAnimateV5-12b-zh-InP"
    # "Inpaint" or "Control"
    model_type = "Inpaint"
    # Save dir
    savedir_sample = "samples"

    # Set up GPU memory management
    clear_gpu_memory()
    check_gpu_memory()

    if ui_mode == "modelscope":
        demo, controller = ui_modelscope(model_type, edition, config_path, model_name, savedir_sample, GPU_memory_mode, weight_dtype)
    elif ui_mode == "eas":
        demo, controller = ui_eas(edition, config_path, model_name, savedir_sample)
    else:
        demo, controller = ui(GPU_memory_mode, weight_dtype)

    # Move model to CPU if needed
    if hasattr(controller, 'model'):
        controller.model = move_to_cpu_if_needed(controller.model)

    # launch gradio
    app, _, _ = demo.queue(status_update_rate=1).launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=True,
        share=True
    )
    
    # launch api
    infer_forward_api(None, app, controller)
    update_diffusion_transformer_api(None, app, controller)
    update_edition_api(None, app, controller)
    
    # Periodically check and clear GPU memory
    while True:
        time.sleep(60)  # Check every minute
        clear_gpu_memory()
        check_gpu_memory()
        if hasattr(controller, 'model'):
            controller.model = move_to_cpu_if_needed(controller.model)
