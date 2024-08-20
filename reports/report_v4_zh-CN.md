# EasyAnimateV4 Report

在EasyAnimateV4版本中，我们在大约20m的数据上进行了训练，支持图片与视频预测，支持像素值从512x512x144、768x768x144、1024x1024x144到1280x1280x96与不同纵横比的视频生成。另外，我们支持图像到视频的生成与视频到视频的重建。不同显存可以生成的视频大小有：

| GPU memory | 384x672x72 | 384x672x144 | 576x1008x72 | 576x1008x144 | 768x1344x72 | 768x1344x144 | 960x1680x96 |
|----------|----------|----------|----------|----------|----------|----------|----------|
| 12GB | ⭕️ | ⭕️ | ⭕️ | ⭕️ | ❌ | ❌ | ❌ |
| 16GB | ✅ | ✅ | ⭕️ | ⭕️ | ⭕️ | ❌ | ❌ |
| 24GB | ✅ | ✅ | ✅ | ⭕️ | ✅ | ❌ | ❌ |
| 40GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 80GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

✅ 表示它可以在low_gpu_memory_mode＝False下运行，⭕️ 表示它可以在low_gpu_memory_mode＝True下运行，❌ 表示它无法运行。low_gpu_memory_mode=True时，运行速度较慢。

有一些不支持torch.bfloat16的卡型，如2080ti、V100，需要将app.py、predict文件中的weight_dtype修改为torch.float16才可以运行。不同卡型在25step时的生成时间如下：

| GPU | 384x672x72 | 384x672x144 | 576x1008x72 | 576x1008x144 | 768x1344x72 | 768x1344x144 | 960x1680x96 |
|----------|----------|----------|----------|----------|----------|----------|----------|
| A10 24GB | ~180s | ~370s | ~480s | ~1800s(⭕️) | ~1000s | ❌ | ❌ |
| A100 80GB | ~60s | ~180s | ~200s | ~600s | ~500s | ~1800s | ~1800s |

除了EasyAnimateV3中引入的功能外，EasyAnimateV4还突出了以下功能：

- 3D Full Attention
- 中文、英文提示词预测
- 基于Token长度的模型训练。达成不同大小多分辨率在同一模型中的实现
- 更多数据和更好的多阶段训练
- 重新训练的VAE

限制：

我们尝试将EasyAnimate以3D full attention进行实现，但该结构在slice vae上表现一般，且训练成本较大，因此V4版本性能并未完全领先V3。由于资源有限，我们正在将EasyAnimate迁移到重新训练的16通道magvit上以追求更好的模型性能。

上述改进的所有实现（包括训练和推理）都可以在EasyAnimateV4版本中获得。以下部分将介绍改进的细节。我们还优化了我们的代码库和文档，使其更易于使用和开发。
## 3D Full Attention

我们使用了[Hunyuan-DiT](https://github.com/Tencent/HunyuanDiT)作为基础结构，并在此基础上修改了VAE和DiT的模型结构来更好地支持视频的生成。

EasyAnimateV4包括两个Text Encoder、Video VAE（视频编码器和视频解码器）和Diffusion Transformer（DiT）。MT5 Encoder和多摸CLIP用作文本编码器。EasyAnimateV4使用3D全局注意力进行视频重建，与V3相比不再划分运动模块与基础模型，直接通过全局注意力确保生成连贯的帧和无缝的运动过渡。

EasyAnimateV4的Pipeline结构如下：

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/framework_v4.jpg" alt="ui" style="zoom:50%;" />

EasyAnimateV4基础模型结构如下：

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/pipeline_v4.jpg" alt="ui" style="zoom:50%;" />

## 中文、英文提示词预测。

高效的文本编码器在文本到图像与文本到视频的生成中至关重要，因为生成任务需要准确理解和编码输入文本提示以生成相应的图像。在当前已有的扩散模型中，CLIP和T5是大多数模型的主流选择。[PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)仅使用T5对输入文本提示的理解，而[Stable Diffusion](https://github.com/runwayml/stable-diffusion)仅仅采用CLIP进行文本编码。一些模型，如[Swinv2-Imagen](https://arxiv.org/abs/2210.09549)融合了CLIP和T5两种编码器，以进一步提高其文本理解能力。

我们使用了[Hunyuan-DiT](https://github.com/Tencent/HunyuanDiT)作为基础结构，选择在文本编码中结合T5和CLIP，以利用这两种模型的优势，从而提高文本到图像生成过程的准确性和多样性。由于使用的T5和CLIP是多模态双语的，我们的模型同时支持中文预测和英文预测。

## 基于Token长度的模型训练

EasyAnimateV4的训练分为多个阶段，除了图片Adapt VAE的阶段外，其它阶段均为视频训练，分别对应了不同的Token长度。

### 1. 图片VAE的适配
我们使用了12M的数据，包括10M的[SAM](https://www.semanticscholar.org/paper/Segment-Anything-Kirillov-Mintun/7470a1702c8c86e6f28d32cfa315381150102f5b)和2M的[JourneyDB](https://github.com/JourneyDB/JourneyDB)进行VAE的适配，总共训练约30K步。

训练时我们采用高低分辨率结合训练，因此模型支持从512到1280任意分辨率的图片生成。

### 2. 视频训练
视频训练则根据不同Token长度，对视频进行缩放后进行训练。

视频训练分为多个阶段，每个阶段的Token长度分别是9216（对应256x256x144的视频），36864（对应512x512x144的视频），82944（对应768x768x144的视频），147456（对应1024x1024x144的视频）。

其中：
- 9216阶段使用了全部大约20M的数据进行训练，Batch size为1024，训练步数为60k。
- 36864阶段使用了720P以上的视频训练（大约7M），Batch size为192，训练步数为16k
- 82944使用了最高质量的视频训练（大约0.8M），Batch size为192，训练步数为5k。
- 147456使用了最高质量的视频训练（大约0.8M），Batch size为96，训练步数为5k。

训练时我们采用高低分辨率结合训练，因此模型支持从512到1280任意分辨率的视频生成，以9216 token长度为例：
- 在256x256分辨率下，视频帧数为144；
- 在512x512分辨率下，视频帧数为32；
- 在1024x1024分辨率下，视频帧数为8；
这些分辨率与对应长度混合训练，模型可以完成不同大小分辨率的视频生成。

## 重新训练的VAE

Slice VAE在面对画面变动时存在一定的顿挫感，因为后面的latent在解码的时候无法看到完全看到前面的块的信息。

参考magvit，我们对前面块卷积后的结果进行了存储，除去最开始的视频块，后面每一个视频块在卷积时，都只能看到前面视频块的特征，看不到后面视频块的特征，在这样的修改后，Decoder的重建结果相比原Slice VAE会更平滑。

我们在上述修改后对Slice VAE进行了重新训练，在批次大小为64的情况下，重新训练了100k步。