import time

import torch

from easyanimate.api.api import (infer_forward_api,
                                 update_diffusion_transformer_api,
                                 update_edition_api)
from easyanimate.ui.ui import ui, ui_eas, ui_modelscope

if __name__ == "__main__":
    # Choose the ui mode  
    ui_mode = "normal"
    
    # GPU memory mode, which can be choosen in ["model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"].
    # "model_cpu_offload" means that the entire model will be moved to the CPU after use, which can save some GPU memory.
    # 
    # "model_cpu_offload_and_qfloat8" indicates that the entire model will be moved to the CPU after use, 
    # and the transformer model has been quantized to float8, which can save more GPU memory. 
    # 
    # "sequential_cpu_offload" means that each layer of the model will be moved to the CPU after use, 
    # resulting in slower speeds but saving a large amount of GPU memory.
    # 
    # EasyAnimateV1, V2 and V3 support "model_cpu_offload" "sequential_cpu_offload"
    # EasyAnimateV4, V5 and V5.1 support "model_cpu_offload" "model_cpu_offload_and_qfloat8" "sequential_cpu_offload"
    GPU_memory_mode = "model_cpu_offload_and_qfloat8"
    # EasyAnimateV5.1 support TeaCache.
    enable_teacache     = True
    # Recommended to be set between 0.05 and 0.1. A larger threshold can cache more steps, speeding up the inference process, 
    # but it may cause slight differences between the generated content and the original content.
    teacache_threshold  = 0.08
    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16

    # Server ip
    server_name = "0.0.0.0"
    server_port = 7860

    # Params below is used when ui_mode = "modelscope"
    edition = "v5.1"
    # Config
    config_path = "config/easyanimate_video_v5.1_magvit_qwen.yaml"
    # Model path of the pretrained model
    model_name = "models/Diffusion_Transformer/EasyAnimateV5.1-12b-zh-InP"
    # "Inpaint" or "Control"
    model_type = "Inpaint"
    # Save dir
    savedir_sample = "samples"

    if ui_mode == "modelscope":
        demo, controller = ui_modelscope(model_type, edition, config_path, model_name, savedir_sample, GPU_memory_mode, enable_teacache, teacache_threshold, weight_dtype)
    elif ui_mode == "eas":
        demo, controller = ui_eas(edition, config_path, model_name, savedir_sample)
    else:
        demo, controller = ui(GPU_memory_mode, enable_teacache, teacache_threshold, weight_dtype)

    # launch gradio
    app, _, _ = demo.queue(status_update_rate=1).launch(
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=True
    )
    
    # launch api
    infer_forward_api(None, app, controller)
    update_diffusion_transformer_api(None, app, controller)
    update_edition_api(None, app, controller)
    
    # not close the python
    while True:
        time.sleep(5)