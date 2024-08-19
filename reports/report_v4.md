## EasyAnimateV4 Report
In the EasyAnimateV4, we trained on approximately 20 million data points, supporting predictions for both images and videos. The model accommodates pixel values ranging from 512x512x144, 768x768x144, 1024x1024x144, to 1280x1280x96, facilitating video generation with different aspect ratios. Additionally, we support image-to-video generation and video-to-video reconstruction. The sizes of videos that can be generated with different GPU memory configurations are as follows:

| GPU memory | 384x672x72 | 384x672x144 | 576x1008x72 | 576x1008x144 | 768x1344x72 | 768x1344x144 | 960x1680x96 |
|----------|----------|----------|----------|----------|----------|----------|----------|
| 12GB | ⭕️ | ⭕️ | ⭕️ | ⭕️ | ❌ | ❌ | ❌ |
| 16GB | ✅ | ✅ | ⭕️ | ⭕️ | ⭕️ | ❌ | ❌ |
| 24GB | ✅ | ✅ | ✅ | ⭕️ | ✅ | ❌ | ❌ |
| 40GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 80GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
✅ indicates it can run with low_gpu_memory_mode=False, ⭕️ indicates it can run with low_gpu_memory_mode=True, and ❌ indicates it cannot run. When low_gpu_memory_mode=True, the speed is slower.

Some GPU models, such as the 2080ti and V100, do not support torch.bfloat16 and require modifying the weight_dtype in the app.py and predict files to torch.float16 in order to operate. The generation times for different GPU models at 25 steps are as follows:

| GPU | 384x672x72 | 384x672x144 | 576x1008x72 | 576x1008x144 | 768x1344x72 | 768x1344x144 | 960x1680x96 |
|----------|----------|----------|----------|----------|----------|----------|----------|
| A10 24GB | ~180s | ~370s | ~480s | ~1800s(⭕️) | ~1000s | ❌ | ❌ |
| A100 80GB | ~60s | ~180s | ~200s | ~600s | ~500s | ~1800s | ~1800s |

In addition to the features introduced in EasyAnimateV3, EasyAnimateV4 also highlights the following functionalities:

- 3D Full Attention
- Prediction with prompts in both Chinese and English
- Model training based on token length, achieving implementation of different sizes and resolutions within a single model
- More data and improved multi-stage training
- Re-trained VAE

Limitation: 

We attempted to implement EasyAnimate using 3D full attention, but this structure performed moderately on slice VAE and incurred considerable training costs. As a result, the performance of version V4 did not significantly surpass that of version V3. Due to limited resources, we are migrating EasyAnimate to a retrained 16-channel MagVit to pursue better model performance.

All implementations of the above improvements (including training and inference) are available in the EasyAnimateV4 version. The following sections will detail the improvements. We have also optimized our codebase and documentation for easier usage and development.

## 3D Full Attention
We utilized [Hunyuan-DiT](https://github.com/Tencent/HunyuanDiT) as our foundational architecture and modified the VAE and DiT model structures to better support video generation. 

EasyAnimateV4 includes two text encoders, Video VAE (video encoder and decoder), and Diffusion Transformer (DiT). The MT5 Encoder and multi-modal CLIP are used as text encoders. EasyAnimateV4 employs 3D global attention for video reconstruction, eliminating the separation of motion modules and base models as seen in V3. This ensures coherent frame generation and seamless motion transitions through global attention.

The pipeline structure of EasyAnimateV4 is as follows:

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/framework_v4.jpg" alt="ui" style="zoom:50%;" />

The foundational model structure of EasyAnimateV4 is as follows:

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/pipeline_v4.jpg" alt="ui" style="zoom:50%;" />

## Prediction with Prompts in Chinese and English
An efficient text encoder is crucial for generating from text-to-image and text-to-video tasks as these tasks require accurate understanding and encoding of input text prompts to generate corresponding images. Among existing diffusion models, CLIP and T5 are mainstream choices for most models. PixArt-alpha solely uses T5 for understanding input text prompts, while Stable Diffusion employs CLIP for text encoding. Some models, like Swinv2-Imagen, integrate both CLIP and T5 encoders to further enhance their text comprehension capabilities.

We adopted Hunyuan-DiT as our foundational architecture and chose to integrate T5 and CLIP in text encoding to leverage the strengths of both models, thereby improving the accuracy and diversity of the text-to-image generation process. Since T5 and CLIP are both multilingual, our model supports predictions in both Chinese and English.

## Token-Length Based Model Training
Training in EasyAnimateV4 is conducted in multiple stages; aside from the image Adapt VAE stage, all other stages focus on video training, corresponding to different token lengths.

### 1. Image VAE Adaptation
We utilized 12 million data points, including 10 million from SAM and 2 million from JourneyDB for the adaptation of the VAE, training approximately 30K steps in total. During training, we incorporated both high and low resolutions, thus allowing the model to generate images at resolutions ranging from 512 to 1280.

### 2. Video Training
Video training is conducted based on various token lengths by scaling the videos accordingly. The video training is divided into multiple stages, with each stage corresponding to specific token lengths: 9216 (for 256x256x144 videos), 36864 (for 512x512x144 videos), 82944 (for 768x768x144 videos), and 147456 (for 1024x1024x144 videos).

Details are as follows:

- The 9216 stage used all approximately 20 million data for training, with a batch size of 1024 and a training step count of 60,000.
- The 36864 stage trained on 720P videos or above (approximately 7 million), with a batch size of 192 for 16,000 training steps.
- The 82944 stage employed high-quality videos (approximately 0.8 million) with a batch size of 192 for 5,000 training steps.
- The 147456 stage likewise utilized high-quality videos (approximately 0.8 million) with a batch size of 96 for 5,000 training steps.

During training, we incorporated both high and low resolutions, thus allowing the model to generate videos at resolutions ranging from 512 to 1280. For instance, at a token length of 9216:
- At a resolution of 256x256, the video frame count is 144;
- At a resolution of 512x512, the video frame count is 32;
- At a resolution of 1024x1024, the video frame count is 8.
These resolutions are mixed during training, enabling the model to generate videos of different sizes and resolutions.

## Re-trained VAE
The Slice VAE exhibits some stuttering during scene changes because the later latents cannot fully access information from the preceding blocks during decoding. 

Referring to MagVit, we stored the results after convolution of the previous blocks. Except for the initial video block, each subsequent video block during convolution only accessed the features of the preceding video blocks, not the following ones. After this modification, the decoder's reconstruction results are smoother compared to the original Slice VAE.

Following this modification, we re-trained the Slice VAE, completing 100,000 training steps with a batch size of 64.