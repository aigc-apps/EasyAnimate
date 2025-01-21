# EasyAnimateV5 Report

In the EasyAnimateV5.1 version, we have replaced the original dual text encoders with Alibaba's recently released Qwen2 VL. Since Qwen2 VL is a multilingual model, EasyAnimateV5.1 supports multilingual predictions, and the language support range is linked to Qwen2 VL. In general, the experience is best with Chinese and English, while Japanese, Korean, and other languages are also supported.

In addition to text-to-video, image-to-video, video-to-video, and general control, we now support trajectory control and camera lens control. With trajectory control, you can manage the specific movement direction of an object, and with camera lens control, you can control the movement of the video camera lens. By combining multiple camera movements, deviations such as left-up and left-down can be achieved.

Compared to EasyAnimateV5, EasyAnimateV5.1 mainly highlights the following features:

- Utilizes Qwen2 VL as the text encoder, supporting multilingual predictions;
- Supports new control methods, such as trajectory control and camera control;
- Optimizes performance using reward algorithms;
- Uses Flow as the sampling method;
- Trains with more data.

## Utilizing Qwen2 VL as the Text Encoder
Based on the MMDiT structure, we replaced EasyAnimateV5's dual text encoders with Alibaba's recently released Qwen2 VL. Compared to CLIP and T5, [Qwen2 VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) outperforms both as a generation and encoding model, offering more precise semantic understanding. Additionally, aligning with images allows Qwen2 VL to have a more accurate understanding of image content compared to Qwen2 itself.

We extract the penultimate feature of Qwen2 VL's hidden_states and input it into MMDiT, performing self-attention with video embedding. Before self-attention, we apply an RMSNorm for value correction and then fully connect, as deep feature values of large language models are generally large (up to tens of thousands),

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/qwen2_vl_transformer.jpg" alt="ui" style="zoom:50%;" />

## Supporting New Control Methods such as Trajectory Control and Camera Control
Referring to [Drag Anything](https://github.com/showlab/DragAnything), we implemented 2D trajectory control, adding trajectory control points before Conv in to manage the movement direction of objects. The Gaussian blur is used to specify the movement direction of objects.

Referring to [CameraCtrl](https://github.com/hehao13/CameraCtrl), we input the control trajectory of the camera lens before Conv in, which determines the direction of lens movement, achieving control of the video lens.

We have implemented corresponding control schemes in [EasyAnimate ComfyUI](../comfyui/README.md), thanks to the node implementations from [KJ Nodes](https://github.com/kijai/ComfyUI-KJNodes) and [ComfyUI-CameraCtrl-Wrapper](https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper).

## Optimizing Performance Using Reward Algorithms
To further enhance the quality of generated videos and better align them with human preferences, we applied reward backpropagation ([DRaFT](https://arxiv.org/abs/2309.17400) and [DRTune](https://arxiv.org/abs/2405.00760)) for further training the base model of EasyAnimateV5.1, using rewards to improve the model's text consistency and image detail.

For details on using reward backpropagation, refer to [EasyAnimate ComfyUI](../scripts/README_TRAIN_REWARD.md).

## Using Flow Matching as the Sampling Method
Beyond the architectural changes mentioned, EasyAnimateV5.1 also adopts the [flow-matching](https://arxiv.org/html/2403.03206v1#S3) approach for training. In this method, the forward noise process is defined as rectifying along a straight line connecting the data and noise distributions.

The corrected flow-matching sampling process is simpler and performs well in reducing sampling steps. Our new scheduler (FlowMatchEulerDiscreteScheduler), consistent with [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium/), includes the corrected flow-matching formula and Euler method steps.

## Training with More Data
Compared to EasyAnimateV5, EasyAnimateV5.1 added about 10M high-resolution data for training.

EasyAnimateV5.1 training consists of multiple phases, with all phases being video training except for the image Adapt VAE phase, corresponding to different Token lengths.

### 1. Image VAE Alignment
We used 10M [SAM](https://www.semanticscholar.org/paper/Segment-Anything-Kirillov-Mintun/7470a1702c8c86e6f28d32cfa315381150102f5b) to train the model from scratch for text-image alignment, training a total of about 120K steps.

Upon completion, the model can generate corresponding images based on prompts, with the targets in the images generally matching the prompt descriptions.

### 2. Video Training
Video training involves scaling videos according to different Token lengths.

Video training is divided into multiple stages, with Token lengths of 3328 (corresponding to 256x256x49 videos), 13312 (corresponding to 512x512x49 videos), and 53248 (corresponding to 1024x1024x49 videos).

Among them:
- 3328 stage
  - Used all data (about 36.6M) to train text-to-video models, with a batch size of 1024, training about 100K steps.
- 13312 stage
  - Used videos above 720P (about 27.9M) to train text-to-video models, with a batch size of 512, training about 60K steps.
  - Used the highest quality videos (about 0.5M) to train image-to-video models, with a batch size of 256, training about 5K steps.
- 53248 stage
  - Used the highest quality videos (about 0.5M) to train image-to-video models, with a batch size of 256, training about 5K steps.

Combining high and low resolution training, the model supports generating videos at any resolution from 512 to 1024. 

For different resolutions at 13312 token length:
- At 512x512 resolution, the video frame count is 49;
- At 768x768 resolution, the video frame count is 21;
- At 1024x1024 resolution, the video frame count is 9;

These resolutions and corresponding lengths are mixed during training, allowing the model to generate videos of varying sizes and resolutions.
