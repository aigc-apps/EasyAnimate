# EasyAnimateV5 Report

在EasyAnimateV5.1版本中，我们将原来的双text encoders替换成alibaba近期发布的Qwen2 VL，由于Qwen2 VL是一个多语言模型，EasyAnimateV5.1支持多语言预测，语言支持范围与Qwen2 VL挂钩，综合体验下来是中文英文最佳，同时还支持日语、韩语等语言的预测。

另外在文生视频、图生视频、视频生视频和通用控制的基础上，我们支持了轨迹控制与相机镜头控制，通过轨迹控制可以实现控制某一物体的具体的运动方向，通过相机镜头控制可以控制视频镜头的运动方向，组合多个镜头运动后，还可以往左上、左下等方向进行偏转。

对比EasyAnimateV5，EasyAnimateV5.1主要突出了以下特点：

- 应用Qwen2 VL作为文本编码器，支持多语言预测；
- 支持轨迹控制，相机控制等新控制方式；
- 使用奖励算法最终优化性能；
- 使用Flow作为采样方式；
- 使用更多数据训练。

## 应用Qwen2 VL作为文本编码器
在MMDiT结构的基础上，我们将EasyAnimateV5的双text encoders替换成alibaba近期发布的Qwen2 VL；相比于CLIP与T5，无论是作为生成模型还是编码模型，[Qwen2 VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)性能更加优越，对语义理解更加精准。而相比于Qwen2本身，Qwen2 VL由于和图像做过对齐，对图像内容的理解更为精确。

我们取出Qwen2 VL hidden_states的倒数第二个特征输入到MMDiT中，与视频Embedding一起做Self-Attention。在做Self-Attention前，由于大语言模型深层特征值一般较大（可以达到几万），我们为其做了一个RMSNorm进行数值的矫正再进行全链接，

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/qwen2_vl_transformer.jpg" alt="ui" style="zoom:50%;" />

## 支持轨迹控制，相机控制等新控制方式
参考[Drag Anything](https://github.com/showlab/DragAnything)，我们使用了2D的轨迹控制，在Conv in前添加轨迹控制点，通过轨迹控制点控制物体的运动方向。通过白色高斯图的方式，指定物体的运动方向。

参考[CameraCtrl](https://github.com/hehao13/CameraCtrl)，我们在Conv in前输入了相机镜头的控制轨迹，该轨迹规定的镜头的运动方向，实现了视频镜头的控制。

我们在[EasyAnimate ComfyUI](../comfyui/README.md)中实现了对应的控制方案，感谢[KJ Nodes](https://github.com/kijai/ComfyUI-KJNodes)和[ComfyUI-CameraCtrl-Wrapper](https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper)中对控制节点的实现。

## 使用奖励算法最终优化性能
为了进一步优化生成视频的质量以更好地对齐人类偏好，我们采用奖励反向传播（[DRaFT](https://arxiv.org/abs/2309.17400) 和 [DRTune](https://arxiv.org/abs/2405.00760)）对 EasyAnimateV5.1 基础模型进行后训练，使用奖励提升模型的文本一致性与画面的精细程度。

我们在[EasyAnimate ComfyUI](../scripts/README_TRAIN_REWARD.md)中详细说明了奖励反向传播的使用方案。

## 使用Flow matching作为采样方式
除了上述的架构变化之外，EasyAnimateV5.1还应用[flow-matching](https://arxiv.org/html/2403.03206v1#S3)的方案来训练模型。在这种方法中，前向噪声过程被定义为在直线上连接数据和噪声分布的整流。

修正流匹配采样过程更简单，在减少采样步骤数时表现良好。与[Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium/)一致，我们新的调度程序（FlowMatchEulerDiscreteScheduler）作为调度器，其中包含修正流匹配公式和欧拉方法步骤。

## 使用更多数据训练
相比于EasyAnimateV5，EasyAnimateV5.1在训练时，我们添加了约10M的高分辨率数据。

EasyAnimateV5.1的训练分为多个阶段，除了图片Adapt VAE的阶段外，其它阶段均为视频训练，分别对应了不同的Token长度。

### 1. 图片VAE的对齐
我们使用了10M的[SAM](https://www.semanticscholar.org/paper/Segment-Anything-Kirillov-Mintun/7470a1702c8c86e6f28d32cfa315381150102f5b)进行模型从0开始的文本图片对齐的训练，总共训练约120K步。

在训练完成后，模型已经有能力根据提示词去生成对应的图片，并且图片中的目标基本符合提示词描述。

### 2. 视频训练
视频训练则根据不同Token长度，对视频进行缩放后进行训练。

视频训练分为多个阶段，每个阶段的Token长度分别是3328（对应256x256x49的视频），13312（对应512x512x49的视频），53248（对应1024x1024x49的视频）。

其中：
- 3328阶段
  - 使用了全部的数据（大约36.6M）训练文生视频模型，Batch size为1024，训练步数约为100k。
- 13312阶段
  - 使用了720P以上的视频训练（大约27.9M）训练文生视频模型，Batch size为512，训练步数约为60k
  - 使用了最高质量的视频训练（大约0.5M）训练图生视频模型 ，Batch size为256，训练步数为5k
- 53248阶段
  - 使用了最高质量的视频训练（大约0.5M）训练图生视频模型，Batch size为256，训练步数为5k。

训练时我们采用高低分辨率结合训练，因此模型支持从512到1024任意分辨率的视频生成，以13312 token长度为例：
- 在512x512分辨率下，视频帧数为49；
- 在768x768分辨率下，视频帧数为21；
- 在1024x1024分辨率下，视频帧数为9；
这些分辨率与对应长度混合训练，模型可以完成不同大小分辨率的视频生成。