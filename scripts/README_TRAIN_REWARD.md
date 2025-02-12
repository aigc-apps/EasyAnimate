# Enhance EasyAnimate with Reward Backpropagation (Preference Optimization)
We explore the Reward Backpropagation technique <sup>[1](#ref1) [2](#ref2)</sup> to optimized the generated videos by [EasyAnimateV5](https://github.com/aigc-apps/EasyAnimate/tree/main/easyanimate) for better alignment with human preferences.
We provide pre-trained models (i.e. LoRAs) along with the training script. You can use these LoRAs to enhance the corresponding base model as a plug-in or train your own reward LoRA.

> [!NOTE]
> For EasyAnimateV5.1, we have merged the reward LoRAs into the base model. Please use the base model directly.

- [Enhance EasyAnimate with Reward Backpropagation (Preference Optimization)](#enhance-easyanimate-with-reward-backpropagation-preference-optimization)
  - [Demo](#demo)
    - [EasyAnimateV5-12b-zh-InP](#easyanimatev5-12b-zh-inp)
    - [EasyAnimateV5-7b-zh-InP](#easyanimatev5-7b-zh-inp)
  - [Model Zoo](#model-zoo)
  - [Inference](#inference)
  - [Training](#training)
    - [Setup](#setup)
    - [Important Args](#important-args)
  - [Limitations](#limitations)
  - [References](#references)


## Demo
### EasyAnimateV5-12b-zh-InP

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
    <thead>
        <tr>
            <th style="text-align: center;" width="10%">Prompt</sup></th>
            <th style="text-align: center;" width="30%">EasyAnimateV5-12b-zh-InP</th>
            <th style="text-align: center;" width="30%">EasyAnimateV5-12b-zh-InP <br> HPSv2.1 Reward LoRA</th>
            <th style="text-align: center;" width="30%">EasyAnimateV5-12b-zh-InP <br> MPS Reward LoRA</th>
        </tr>
    </thead>
    <tr>
        <td>
            Porcelain rabbit hopping by a golden cactus
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/c7ee83b2-0329-4853-b47d-e8e1550f1164" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/1fea5b95-05dd-44cf-aec2-5c104e3afa8d" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/de14593a-daae-4a3e-8231-7df2108065d5" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            Yellow rubber duck floating next to a blue bath towel
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/c146fe30-ddcc-4e26-8659-885efd48136f" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/bd4a0a5c-cfe0-4a04-835b-1a3613926a6d" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/f5076984-9661-4670-9ca5-abc33b7d66c0" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            An elephant sprays water with its trunk, a lion sitting nearby
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/139bc722-d8bb-42cb-b043-99334f320496" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/87edf580-f1f3-4be2-931e-e53306ca9087" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/a38581c2-f4b3-4905-93af-debb3aec6488" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            A fish swims gracefully in a tank as a horse gallops outside
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/0383cdd5-1d9c-4b62-bde9-7a0423c8f863" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/efaee3eb-c361-4167-8952-92853a13df24" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/4cd406e3-8348-4589-8c07-43379547e1e1" width="100%" controls autoplay loop></video>
        </td>
    </tr>
</table>

### EasyAnimateV5-7b-zh-InP

<table border="0" style="width: 100%; text-align: center; margin-top: 20px;">
    <thead>
        <tr>
            <th style="text-align: center;" width="10%">Prompt</th>
            <th style="text-align: center;" width="30%">EasyAnimateV5-7b-zh-InP</th>
            <th style="text-align: center;" width="30%">EasyAnimateV5-7b-zh-InP <br> HPSv2.1 Reward LoRA</th>
            <th style="text-align: center;" width="30%">EasyAnimateV5-7b-zh-InP <br> MPS Reward LoRA</th>
        </tr>
    </thead>
    <tr>
        <td>
            Crystal cake shimmering beside a metal apple
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/25ae8abe-2e53-4557-b3f0-a72c247603e2" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/26f47c9b-e8f6-4768-978f-56fb47de4f2f" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/56166d66-4645-409e-b236-48ea25e8400b" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            Elderly artist with a white beard painting on a white canvas
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/7e0d7153-036a-4a40-b726-218760837ce7" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/314a68e8-57e3-437e-9acc-656da5f73853" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/d045e3e8-c9bd-4833-9a00-6decd50047d9" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            Porcelain rabbit hopping by a golden cactus
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/93890751-2ae7-4d55-82dc-7f992c8ad9b4" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/932ef7e4-c8a9-4153-94a8-8975d872701e" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/be0a01aa-a0c7-45a1-9db2-3b718c0be272" width="100%" controls autoplay loop></video>
        </td>
    </tr>
    <tr>
        <td>
            Green parrot perching on a brown chair
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/74a41dd4-8375-44be-8242-11287037c484" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/fd76e645-4ae3-427f-ac7b-9712e6dae4dd" width="100%" controls autoplay loop></video>
        </td>
        <td>
            <video src="https://github.com/user-attachments/assets/6a7a0c11-1a78-4d51-90c4-814d1f4fb338" width="100%" controls autoplay loop></video>
        </td>
    </tr>
</table>

> [!NOTE]
> The above test prompts are from <a href="https://github.com/KaiyueSun98/T2V-CompBench">T2V-CompBench</a>. All videos are generated with lora weight 0.7.

## Model Zoo
| Name | Base Model | Reward Model | Hugging Face | Description |
|--|--|--|--|--|
| EasyAnimateV5-12b-zh-InP-HPS2.1.safetensors | EasyAnimateV5-12b-zh-InP | [HPS v2.1](https://github.com/tgxs002/HPSv2) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs/resolve/main/EasyAnimateV5-12b-zh-InP-HPS2.1.safetensors) | Official HPS v2.1 reward LoRA (`rank=128` and `network_alpha=64`) for EasyAnimateV5-12b-zh-InP. It is trained with a batch size of 8 for 2,500 steps.|
| EasyAnimateV5-7b-zh-InP-HPS2.1.safetensors | EasyAnimateV5-7b-zh-InP | [HPS v2.1](https://github.com/tgxs002/HPSv2) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs/resolve/main/EasyAnimateV5-7b-zh-InP-HPS2.1.safetensors) | Official HPS v2.1 reward LoRA (`rank=128` and `network_alpha=64`) for EasyAnimateV5-7b-zh-InP. It is trained with a batch size of 8 for 3,500 steps.|
| EasyAnimateV5-12b-zh-InP-MPS.safetensors | EasyAnimateV5-12b-zh-InP | [MPS](https://github.com/Kwai-Kolors/MPS) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs/resolve/main/EasyAnimateV5-12b-zh-InP-MPS.safetensors) | Official MPS reward LoRA (`rank=128` and `network_alpha=64`) for EasyAnimateV5-12b-zh-InP. It is trained with a batch size of 8 for 2,500 steps.|
| EasyAnimateV5-7b-zh-InP-MPS.safetensors | EasyAnimateV5-7b-zh-InP | [MPS](https://github.com/Kwai-Kolors/MPS) | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs/resolve/main/EasyAnimateV5-7b-zh-InP-MPS.safetensors) | Official MPS reward LoRA (`rank=128` and `network_alpha=64`) for EasyAnimateV5-7b-zh-InP. It is trained with a batch size of 8 for 2,000 steps.|

## Inference
We provide an example inference code to run EasyAnimateV5-12b-zh-InP with its HPS2.1 reward LoRA.

```python
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer

from easyanimate.models import AutoencoderKLMagvit, EasyAnimateTransformer3DModel
from easyanimate.pipeline.pipeline_easyanimate_inpaint import EasyAnimateInpaintPipeline
from easyanimate.utils.lora_utils import merge_lora
from easyanimate.utils.utils import get_image_to_video_latent, save_videos_grid
from easyanimate.utils.fp8_optimization import convert_weight_dtype_wrapper

# GPU memory mode, which can be choosen in [model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
GPU_memory_mode = "model_cpu_offload"
# Download from https://raw.githubusercontent.com/aigc-apps/EasyAnimate/refs/heads/main/config/easyanimate_video_v5_magvit_multi_text_encoder.yaml
config_path = "config/easyanimate_video_v5_magvit_multi_text_encoder.yaml"
model_path = "alibaba-pai/EasyAnimateV5-12b-zh-InP"
lora_path = "alibaba-pai/EasyAnimateV5-Reward-LoRAs/EasyAnimateV5-12b-zh-InP-HPS2.1.safetensors"
weight_dtype = torch.bfloat16
lora_weight = 0.7

prompt = "A panda eats bamboo while a monkey swings from branch to branch"
sample_size = [512, 512]
video_length = 49

config = OmegaConf.load(config_path)
transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
if weight_dtype == torch.float16:
    transformer_additional_kwargs["upcast_attention"] = True
transformer = EasyAnimateTransformer3DModel.from_pretrained_2d(
    model_path, 
    subfolder="transformer",
    transformer_additional_kwargs=transformer_additional_kwargs,
    torch_dtype=torch.float8_e4m3fn if GPU_memory_mode == "model_cpu_offload_and_qfloat8" else weight_dtype,
    low_cpu_mem_usage=True,
)
vae = AutoencoderKLMagvit.from_pretrained(
    model_path, subfolder="vae", vae_additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])
).to(weight_dtype)
if config['vae_kwargs'].get('vae_type', 'AutoencoderKL') == 'AutoencoderKLMagvit' and weight_dtype == torch.float16:
    vae.upcast_vae = True

pipeline = EasyAnimateInpaintPipeline.from_pretrained(
    model_path,
    text_encoder=BertModel.from_pretrained(model_path, subfolder="text_encoder").to(weight_dtype),
    text_encoder_2=T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder_2").to(weight_dtype),
    tokenizer=BertTokenizer.from_pretrained(model_path, subfolder="tokenizer"),
    tokenizer_2=T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer_2"),
    vae=vae,
    transformer=transformer,
    scheduler=DDIMScheduler.from_pretrained(model_path, subfolder="scheduler"),
    torch_dtype=weight_dtype
)
if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload()
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    pipeline.enable_model_cpu_offload()
    for _text_encoder in [pipeline.text_encoder, pipeline.text_encoder_2]:
        if hasattr(_text_encoder, "visual"):
            del _text_encoder.visual
    convert_weight_dtype_wrapper(pipeline.transformer, weight_dtype)
else:
    pipeline.enable_model_cpu_offload()
pipeline = merge_lora(pipeline, lora_path, lora_weight, device="cuda", dtype=weight_dtype)

generator = torch.Generator(device="cuda").manual_seed(42)
input_video, input_video_mask, _ = get_image_to_video_latent(None, None, video_length=video_length, sample_size=sample_size)
sample = pipeline(
    prompt, 
    video_length = video_length,
    negative_prompt = "bad detailed",
    height = sample_size[0],
    width = sample_size[1],
    generator = generator,
    guidance_scale = 7.0,
    num_inference_steps = 50,
    video = input_video,
    mask_video = input_video_mask,
).frames

save_videos_grid(sample, "samples/output.mp4", fps=8)
```

## Training
The [training code](./train_reward_lora.py) is based on [train_lora.py](./train_lora.py).
We provide [a shell script](./train_reward_lora.sh) to train the HPS v2.1 reward LoRA for EasyAnimateV5-12b-zh-InP.

### Setup
Please read the [quick-start](https://github.com/aigc-apps/CogVideoX-Fun/blob/main/README.md#quick-start) section to setup the CogVideoX-Fun environment.
**If you're playing with HPS reward model**, please run the following script to install the dependencies:
```bash
# For HPS reward model only
pip install hpsv2
site_packages=$(python -c "import site; print(site.getsitepackages()[0])")
wget -O $site_packages/hpsv2/src/open_clip/factory.py https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/package/patches/hpsv2_src_open_clip_factory_patches.py
wget -O $site_packages/hpsv2/src/open_clip/ https://github.com/tgxs002/HPSv2/raw/refs/heads/master/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
```

> [!NOTE]
> Since some models will be downloaded automatically from HuggingFace, Please run `HF_ENDPOINT=https://hf-mirror.com sh scripts/train_reward_lora.sh` if you cannot access to huggingface.com.

### Important Args
+ `rank`: The size of LoRA model. The higher the LoRA rank, the more parameters it has, and the more it can learn (including some unnecessary information).
Bt default, we set the rank to 128. You can lower this value to reduce training GPU memory and the LoRA file size.
+ `network_alpha`: A scaling factor changes how the LoRA affect the base model weight. In general, it can be set to half of the `rank`.
+ `prompt_path`: The path to the prompt file (in txt format, each line is a prompt) for sampling training videos. 
We randomly selected 701 prompts from [MovieGenBench](https://github.com/facebookresearch/MovieGenBench/blob/main/benchmark/MovieGenVideoBench.txt).
+ `train_sample_height` and `train_sample_width`: The resolution of the sampled training videos. We found 
training at a 256x256 resolution can generalize to any other resolution. Reducing the resolution can save GPU memory 
during training, but it is recommended that the resolution should be equal to or greater than the image input resolution of the reward model. 
Due to the resize and crop preprocessing operations, we suggest using a 1:1 aspect ratio.
+ `reward_fn` and `reward_fn_kwargs`: The reward model name and its keyword arguments. All supported reward models 
(Aesthetic Predictor [v2](https://github.com/christophschuhmann/improved-aesthetic-predictor)/[v2.5](https://github.com/discus0434/aesthetic-predictor-v2-5), 
[HPS](https://github.com/tgxs002/HPSv2) v2/v2.1, [PickScore](https://github.com/yuvalkirstain/PickScore) and [MPS](https://github.com/Kwai-Kolors/MPS)) 
can be found in [reward_fn.py](../cogvideox/reward/reward_fn.py). 
You can also customize your own reward model (e.g., combining aesthetic predictor with HPS).
+ `num_decoded_latents` and `num_sampled_frames`: The number of decoded latents (for VAE) and sampled frames (for the reward model). 
Since EasyAnimate adopts the 3D casual VAE, we found decoding only the first latent to obtain the first frame for computing the reward 
not only reduces training GPU memory usage but also prevents excessive reward optimization and maintains the dynamics of generated videos.

> [!NOTE]
> In EasyAnimateV5, we only retained the gradient of the last step in the denoising process to reduce GPU memory usage. However, for V5.1, we found that if we only perform reward backpropagation on the last step, the gradient norm becomes very small (usually below 0.001), making it difficult for reward training to converge. This might be due to V5.1 adopts the flow-matching sampling in the training and inference. Therefore, in pratice, we retain the gradients of the last several steps for V5.1.

## Limitations
1. We observe after training to a certain extent, the reward continues to increase, but the quality of the generated videos does not further improve. 
   The model trickly learns some shortcuts (by adding artifacts in the background, i.e., reward hacking) to increase the reward.
2. Currently, there is still a lack of suitable preference models for video generation. Directly using image preference models cannot 
   evaluate preferences along the temporal dimension (such as dynamism and consistency). Further more, We find using image preference models leads to a decrease 
   in the dynamism of generated videos. Although this can be mitigated by computing the reward using only the first frame of the decoded video, the impact still persists.

## References
<ol>
  <li id="ref1">Wu, Xiaoshi, et al. "Deep reward supervisions for tuning text-to-image diffusion models." In ECCV 2025.</li>
  <li id="ref2">Clark, Kevin, et al. "Directly fine-tuning diffusion models on differentiable rewards.". In ICLR 2024.</li>
  <li id="ref3">Prabhudesai, Mihir, et al. "Aligning text-to-image diffusion models with reward backpropagation." arXiv preprint arXiv:2310.03739 (2023).</li>
</ol>