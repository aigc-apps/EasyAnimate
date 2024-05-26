from easyanimate.ui.ui import ui_modelscope, ui

if __name__ == "__main__":
    # Choose the ui mode  
    ui_mode = "normal"
    # Server ip
    server_name = "0.0.0.0"

    # Params below is used when ui_mode = "modelscope"
    edition = "v2"
    config_path = "config/easyanimate_video_magvit_motion_module_v2.yaml"
    model_name = "models/Diffusion_Transformer/EasyAnimateV2-XL-2-512x512"
    savedir_sample = "samples"

    if ui_mode != "modelscope":
        demo = ui()
    else:
        demo = ui_modelscope(edition, config_path, model_name, savedir_sample)
    demo.launch(server_name=server_name)