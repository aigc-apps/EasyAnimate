# EasyAnimateV5 Report

在EasyAnimateV5版本中，我们在大约10m SAM图片数据+26m 图片视频混合的预训练数据上进行了从0开始训练。与之前类似的是，EasyAnimateV5依然支持图片与视频预测与中文英文双语预测，同时支持文生视频、图生视频和视频生视频。

参考[CogVideoX](https://github.com/THUDM/CogVideo/)，我们缩短了视频的总帧数并减少了视频的FPS以加快训练速度，最大支持FPS为8，总长度为49的视频生成。我们支持像素值从512x512x49、768x768x49、1024x1024x49与不同纵横比的视频生成。

对比EasyAnimateV4，EasyAnimateV5还突出了以下特点：

- 应用MMDIT结构，拓展模型规模到12B。
- 支持不同控制输入的控制模型。
- 参考图片添加Noise。
- 更多数据和更好的多阶段训练。

## 应用MMDIT结构，拓展模型规模到12B
参考[Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium/)和[CogVideoX](https://github.com/THUDM/CogVideo/)，在我们的模型中，我们将文本和视频嵌入连接起来以实现Self-Attention，从而更好地对齐视觉和语义信息。然而，这两种模态的特征空间之间存在显著差异，这可能导致它们的数值存在较大差异，这并不利于二者进行对齐。

为了解决这个问题，还是参考[Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium/)，我们采用MMDiT架构作为我们的基础模型，我们为每种模态实现了不同的完全连接结构和前馈网络（FFN），并在一个Self-Attention中实现信息交互，以增强它们的对齐。

另外，为了提高模型的理解能力，我们将模型进行了放大，参考[Flux](https://github.com/black-forest-labs/flux)，我们模型的总参数量扩展到了12B。

## 添加控制信号的EasyAnimateV5
在原本Inpaint模型的基础上，我们使用控制信号替代了原本的mask信号，将控制信号使用VAE编码后作为Guidance与latent一起进入patch流程。该方案已经在[CogVideoX-FUN](https://github.com/aigc-apps/CogVideoX-FUN)的实践中证实有效。

我们在26m的预训练数据中进行了筛选，选择出大约443k高质量视频，同时使用不同的处理方式包含OpenPose、Scribble、Canny、Anime、MLSD、Hed和Depth进行控制条件的提取，作为condition控制信号进行训练。

在进行训练时，我们根据不同Token长度，对视频进行缩放后进行训练。整个训练过程分为两个阶段，每个阶段的13312（对应512x512x49的视频），53248（对应1024x1024x49的视频）。

以EasyAnimateV5-V1.1-5b-Pose为例子，其中：
- 13312阶段，Batch size为128，训练步数为5k
- 53248阶段，Batch size为96，训练步数为2k。

工作原理图如下：
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5/pipeline_control.jpg" alt="ui" style="zoom:50%;" />

## 参考图片添加Noise

在[CogVideoX-FUN](https://github.com/aigc-apps/CogVideoX-FUN)的实践中我们已经发现，在视频生成中，在视频中添加噪声对视频的生成结果有非常大的影响。参考[CogVideoX](https://github.com/THUDM/CogVideo/)和[SVD](https://github.com/Stability-AI/generative-models)，在非0的参考图向上添加Noise以破环原图，追求更大的运动幅度。

我们在模型中为参考图片添加了Noise。与[CogVideoX](https://github.com/THUDM/CogVideo/)一致，在进入VAE前，我们在均值为-3.0、标准差为0.5的正态分布中采样生成噪声幅度，并对其取指数，以确保噪声的幅度在合理范围内。

函数生成与输入视频相同形状的随机噪声，并根据已计算的噪声幅度进行缩放。噪声仅添加到有效值（不需要生成的像素帧上）上，随后与原图像叠加以得到加噪后的图像。

另外，提示词对生成结果影响较大，请尽量描写动作以增加动态性。如果不知道怎么写正向提示词，可以使用smooth motion or in the wind来增加动态性。并且尽量避免在负向提示词中出现motion等表示动态的词汇。

Pipeline结构如下：
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5/pipeline_inpaint.jpg" alt="ui" style="zoom:50%;" />

## 基于Token长度的模型训练

EasyAnimateV5的训练分为多个阶段，除了图片Adapt VAE的阶段外，其它阶段均为视频训练，分别对应了不同的Token长度。

### 1. 图片VAE的对齐
我们使用了10M的[SAM](https://www.semanticscholar.org/paper/Segment-Anything-Kirillov-Mintun/7470a1702c8c86e6f28d32cfa315381150102f5b)进行模型从0开始的文本图片对齐的训练，总共训练约120K步。

在训练完成后，模型已经有能力根据提示词去生成对应的图片，并且图片中的目标基本符合提示词描述。

### 2. 视频训练
视频训练则根据不同Token长度，对视频进行缩放后进行训练。

视频训练分为多个阶段，每个阶段的Token长度分别是3328（对应256x256x49的视频），13312（对应512x512x49的视频），53248（对应1024x1024x49的视频）。

其中：
- 3328阶段
  - 使用了全部的数据（大约26.6M）训练文生视频模型，Batch size为1536，训练步数为66.5k。
- 13312阶段
  - 使用了720P以上的视频训练（大约17.9M）训练文生视频模型，Batch size为768，训练步数为30k
  - 使用了最高质量的视频训练（大约0.5M）训练图生视频模型 ，Batch size为384，训练步数为5k
- 53248阶段
  - 使用了最高质量的视频训练（大约0.5M）训练图生视频模型，Batch size为196，训练步数为5k。

训练时我们采用高低分辨率结合训练，因此模型支持从512到1280任意分辨率的视频生成，以13312 token长度为例：
- 在512x512分辨率下，视频帧数为49；
- 在768x768分辨率下，视频帧数为21；
- 在1024x1024分辨率下，视频帧数为9；
这些分辨率与对应长度混合训练，模型可以完成不同大小分辨率的视频生成。