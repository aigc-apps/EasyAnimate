# EasyAnimateV5 Report

In the EasyAnimateV5 version, we trained from scratch on approximately 10 million SAM image data and 26 million mixed image and video pre-training data. Similar to previous versions, EasyAnimateV5 continues to support image and video prediction as well as bilingual prediction in Chinese and English. Additionally, it supports generating videos from text (text-to-video), images (image-to-video), and other videos (video-to-video).

Referring to [CogVideoX](https://github.com/THUDM/CogVideo/), we shortened the total number of frames and reduced the FPS of the videos to speed up training, with a maximum FPS supported of 8 for videos with a total length of 49 frames. We support video generation with pixel values of 512x512x49, 768x768x49, 1024x1024x49, and different aspect ratios.

Compared to EasyAnimateV4, EasyAnimateV5 also features the following enhancements:
- Application of MMDIT structure, expanding model size to 12B.
- Support for control models with different control inputs.
- Adding Noise to reference images.
- More data and better multi-stage training.

## Application of MMDIT Structure, Expanding Model Size to 12B

Referring to [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium/) and [CogVideoX](https://github.com/THUDM/CogVideo/), in our model, we connected text and video embeddings to achieve Self-Attention, thus better aligning visual and semantic information. However, there is a significant difference in the feature space between these two modalities, which could lead to substantial numerical discrepancies that are not conducive to alignment.

To solve this problem, also referring to [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium/), we adopted the MMDiT architecture as our base model, implementing different fully connected structures and feed-forward networks (FFN) for each modality, and achieving information exchange in one Self-Attention to enhance their alignment.

Additionally, to enhance the model's understanding capability, we upscaled the model. Referring to [Flux](https://github.com/black-forest-labs/flux), the total parameter count of our model expanded to 12B.

## EasyAnimateV5 with Added Control Signals

Based on the original Inpaint model, we replaced the original mask signal with control signals, employing VAE to encode the control signals and include them in the patch process as Guidance together with latents. This scheme has been proven effective in [CogVideoX-FUN](https://github.com/aigc-apps/CogVideoX-FUN).

We screened approximately 443k high-quality videos from 26 million pre-training datasets, using different processing methods including OpenPose, Scribble, Canny, Anime, MLSD, Hed, and Depth to extract control conditions, and trained them as condition control signals.

During training, videos were scaled according to different Token lengths before training. The entire training process was divided into two stages, with each stage being 13312 (corresponding to 512x512x49 videos) and 53248 (corresponding to 1024x1024x49 videos).

Taking EasyAnimateV5-V1.1-5b-Pose as an example, the training procedure is as follows:
- During the 13312 stage, the Batch size is 128, with 5k training steps.
- During the 53248 stage, the Batch size is 96, with 2k training steps.

The working principle is illustrated below:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5/pipeline_control.jpg)

## Adding Noise to Reference Images

In [CogVideoX-FUN](https://github.com/aigc-apps/CogVideoX-FUN) practices, we have found that adding noise to videos significantly impacts the video generation results. Referring to [CogVideoX](https://github.com/THUDM/CogVideo/) and [SVD](https://github.com/Stability-AI/generative-models), noise was added to references with non-zero values to destroy the original image, aiming for greater movement magnitude.

We added noise to the reference images in the model. Consistent with [CogVideoX](https://github.com/THUDM/CogVideo/), before entering VAE, we sampled noise amplitude from a normal distribution with a mean of -3.0 and standard deviation of 0.5, taking exponentials to ensure the noise amplitude is within a reasonable range.

The function generates random noise with the same shape as the input video and scales it according to the pre-computed noise amplitude. The noise is added only to valid values (excluding frames where no generation is needed) and subsequently superimposed on the original image to obtain the noised image.

Additionally, prompts significantly influence the generation results; try to describe actions to increase dynamics. If unsure how to write positive prompts, use terms like "smooth motion" or "in the wind" to increase dynamics. Avoid terms like "motion" indicating dynamics in negative prompts.

The pipeline structure is as follows:
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5/pipeline_inpaint.jpg)

## Model Training Based on Token Length

The training of EasyAnimateV5 is divided into multiple stages. Aside from the image Adapt VAE stage, other stages are video training, each corresponding to a different Token length.

### 1. Alignment of Image VAE

We used 10M [SAM](https://www.semanticscholar.org/paper/Segment-Anything-Kirillov-Mintun/7470a1702c8c86e6f28d32cfa315381150102f5b) for text-image alignment training from scratch, with a total of approximately 120K training steps.

After training, the model is capable of generating corresponding images based on prompts, and the targets in the images generally meet the prompt descriptions.

### 2. Video Training

Video training involves scaling videos according to different Token lengths before training.

Video training is divided into multiple stages, with Token lengths of 3328 (corresponding to 256x256x49 videos), 13312 (corresponding to 512x512x49 videos), and 53248 (corresponding to 1024x1024x49 videos) for each stage.

- For the 3328 stage:
  - All data (approximately 26.6M) was used to train the text-to-video model, with a Batch size of 1536 and 66.5k training steps.

- For the 13312 stage:
  - Videos of 720P and above (approximately 17.9M) were used to train the text-to-video model, with a Batch size of 768 and 30k training steps.
  - The highest quality videos (approximately 0.5M) were used to train the image-to-video model, with a Batch size of 384 and 5k training steps.

- For the 53248 stage:
  - The highest quality videos (approximately 0.5M) were used to train the image-to-video model, with a Batch size of 196 and 5k training steps.

During training, we adopted high-low resolution combined training, so the model supports video generation with arbitrary resolutions from 512 to 1280. Taking 13312 token length as an example:
- At 512x512 resolution, the video frame count is 49.
- At 768x768 resolution, the video frame count is 21.
- At 1024x1024 resolution, the video frame count is 9.

These resolutions are mixed and trained corresponding to their lengths, enabling the model to generate videos of varying resolutions and sizes.
