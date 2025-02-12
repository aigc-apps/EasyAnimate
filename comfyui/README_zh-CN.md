# ComfyUI EasyAnimate
在ComfyUI中使用EasyAnimate!

[![Arxiv Page](https://img.shields.io/badge/Arxiv-Page-red)](https://arxiv.org/abs/2405.18991)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://easyanimate.github.io/)
[![Modelscope Studio](https://img.shields.io/badge/Modelscope-Studio-blue)](https://modelscope.cn/studios/PAI/EasyAnimate/summary)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/EasyAnimate)

[English](./README.md) | 简体中文

- [安装](#安装)
- [节点类型](#节点类型)
- [示例工作流](#示例工作流)
  
## 安装
### 选项1：通过ComfyUI管理器安装
![](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/ComfyUI_Manager.jpg)

### 选项2：手动安装
EasyAnimate存储库需要放置在`ComfyUI/custom_nodes/EasyAnimate/`。

```
cd ComfyUI/custom_nodes/

# Git clone the easyanimate itself
git clone https://github.com/aigc-apps/EasyAnimate.git

# Git clone the video outout node
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
git clone https://github.com/kijai/ComfyUI-KJNodes.git

cd EasyAnimate/
pip install -r comfyui/requirements.txt
```

## 将模型下载到`ComfyUI/models/EasyAnimate/`

EasyAnimateV5.1：

7B:
| 名称 | 种类 | 存储空间 | Hugging Face | Model Scope | 描述 |
|--|--|--|--|--|--|
| EasyAnimateV5.1-7b-zh-InP | EasyAnimateV5.1 | 30 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh-InP) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-7b-zh-InP)| 官方的图生视频权重。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以49帧、每秒8帧进行训练，支持多语言预测 |
| EasyAnimateV5.1-7b-zh-Control | EasyAnimateV5.1 | 30 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh-Control) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-7b-zh-Control)| 官方的视频控制权重，支持不同的控制条件，如Canny、Depth、Pose、MLSD等，同时支持使用轨迹控制。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以49帧、每秒8帧进行训练，支持多语言预测 |
| EasyAnimateV5.1-7b-zh-Control-Camera | EasyAnimateV5.1 | 30 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh-Control-Camera) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-7b-zh-Control-Camera)| 官方的视频相机控制权重，支持通过输入相机运动轨迹控制生成方向。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以49帧、每秒8帧进行训练，支持多语言预测 |
| EasyAnimateV5.1-7b-zh | EasyAnimateV5.1 | 30 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-7b-zh) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5.1-7b-zh)| 官方的文生视频权重。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以49帧、每秒8帧进行训练，支持多语言预测 |

12B：
|名称|类型|存储空间|拥抱面|型号范围|描述|
|--|--|--|--|--|--|
|EasyAnimateV5.1-12b-zh-InP | EasyAnimateV5.1 | 39 GB |[🤗链接](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-InP) | [😄链接](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-InP)|官方图像到视频权重。支持多种分辨率（5127681024）的视频预测，以每秒8帧的速度训练49帧，支持多语言预测|
|EasyAnimateV5.1-12b-zh-控件| EasyAnimateV5.1 | 39 GB |[🤗链接](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control) | [😄链接](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-Control)|官方视频控制权重，支持Canny、Depth、Pose、MLSD和轨迹控制等各种控制条件。支持多种分辨率（5127681024）的视频预测，以每秒8帧的速度训练49帧，支持多语言预测|
|EasyAnimateV5.1-12b-zh-控制摄像头| EasyAnimateV5.1 | 39 GB |[🤗链接](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera) | [😄链接](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-Control-Camera)|官方摄像机控制权重，支持通过输入摄像机运动轨迹进行方向生成控制。支持多种分辨率（5127681024）的视频预测，以每秒8帧的速度训练49帧，支持多语言预测|
|EasyAnimateV5.1-12b-zh| EasyAnimateV5.1 | 39 GB |[🤗链接](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh) | [😄链接](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh)|官方文本到视频权重。支持多种分辨率（5127681024）的视频预测，以每秒8帧的速度训练49帧，支持多语言预测|

<details>
  <summary>(Obsolete) EasyAnimateV5:</summary>

7B:
| 名称 | 种类 | 存储空间 | Hugging Face | Model Scope | 描述 |
|--|--|--|--|--|--|
| EasyAnimateV5-7b-zh-InP | EasyAnimateV5 | 22 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-7b-zh-InP) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-7b-zh-InP)| 官方的7B图生视频权重。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以49帧、每秒8帧进行训练，支持中文与英文双语预测 |
| EasyAnimateV5-7b-zh | EasyAnimateV5 | 22 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-7b-zh) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh)| 官方的7B文生视频权重。可用于进行下游任务的fientune。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以49帧、每秒8帧进行训练，支持中文与英文双语预测 |
| EasyAnimateV5-Reward-LoRAs | EasyAnimateV5 | - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-Reward-LoRAs) | 通过奖励反向传播技术，优化了EasyAnimateV5-12b生成的视频，以更好地匹配人类偏好｜

12B:
| 名称 | 种类 | 存储空间 | Hugging Face | Model Scope | 描述 |
|--|--|--|--|--|--|
| EasyAnimateV5-12b-zh-InP | EasyAnimateV5 | 34 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh-InP) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh-InP)| 官方的图生视频权重。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以49帧、每秒8帧进行训练，支持中文与英文双语预测 |
| EasyAnimateV5-12b-zh-Control | EasyAnimateV5 | 34 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh-Control) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh-Control)| 官方的视频控制权重，支持不同的控制条件，如Canny、Depth、Pose、MLSD等。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以49帧、每秒8帧进行训练，支持中文与英文双语预测 |
| EasyAnimateV5-12b-zh | EasyAnimateV5 | 34 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh)| 官方的文生视频权重。可用于进行下游任务的fientune。支持多分辨率（512，768，1024）的视频预测，支持多分辨率（512，768，1024）的视频预测，以49帧、每秒8帧进行训练，支持中文与英文双语预测 |
| EasyAnimateV5-Reward-LoRAs | EasyAnimateV5 | - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-Reward-LoRAs) | 通过奖励反向传播技术，优化了EasyAnimateV5-12b生成的视频，以更好地匹配人类偏好｜
</details>

<details>
  <summary>(Obsolete) EasyAnimateV4:</summary>

| 名称 | 种类 | 存储空间 | Hugging Face | Model Scope | 描述 |
|--|--|--|--|--|--|
| EasyAnimateV4-XL-2-InP | EasyAnimateV4 | 解压前 8.9 GB / 解压后 14.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV4-XL-2-InP)| [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV4-XL-2-InP)| 官方的图生视频权重。支持多分辨率（512，768，1024，1280）的视频预测，以144帧、每秒24帧进行训练 |
</details>

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>

| 名称 | 种类 | 存储空间 | Hugging Face | Model Scope | 描述 |
|--|--|--|--|--|--|
| EasyAnimateV3-XL-2-InP-512x512 | EasyAnimateV3 | 18.2GB| [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-512x512)| [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-512x512)| 官方的512x512分辨率的图生视频权重。以144帧、每秒24帧进行训练 |
| EasyAnimateV3-XL-2-InP-768x768 | EasyAnimateV3 | 18.2GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-768x768) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-768x768)| 官方的768x768分辨率的图生视频权重。以144帧、每秒24帧进行训练 |
| EasyAnimateV3-XL-2-InP-960x960 | EasyAnimateV3 | 18.2GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-960x960) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-960x960)| 官方的960x960（720P）分辨率的图生视频权重。以144帧、每秒24帧进行训练 |
</details>

## 节点类型
- **LoadEasyAnimateModel**
    - 加载EasyAnimate模型
- **EasyAnimate_TextBox**
    - 编写EasyAnimate模型的提示词
- **EasyAnimateI2VSampler**
    - EasyAnimate图像到视频采样节点
- **EasyAnimateT2VSampler**
    - EasyAnimate文本到视频采样节点
- **EasyAnimateV2VSampler**
    - EasyAnimate视频到视频采样节点

## 示例工作流

### 文本到视频生成
我们的用户界面显示如下，这是[json](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_t2v.json)：

![工作流程图](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_t2v.jpg)

### 图像到视频生成
我们的用户界面显示如下，这是[json](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_i2v.json)：

![工作流程图](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_i2v.jpg)

您可以使用以下照片运行演示：

![演示图像](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/firework.png)

### 视频到视频生成
我们的用户界面显示如下，这是[json](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_v2v.json)：

![工作流程图](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_v2v.jpg)

您可以使用以下视频运行演示：

[演示视频](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/play_guitar.mp4)

### 镜头控制视频生成
我们的用户界面显示如下，这是[json](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_control_camera.json)：

![工作流程图](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_control_camera.jpg)

您可以使用以下照片运行演示：

![演示图像](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/firework.png)

### 轨迹控制视频生成
我们的用户界面显示如下，这是[json](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_control_trajectory.json)：

![工作流程图](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_control_trajectory.jpg)

您可以使用以下照片运行演示：

![演示图像](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/dog.png)

### 控制视频生成
我们的用户界面显示如下，这是[json](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5/easyanimatev5.1_workflow_v2v_control.json)：

![工作流程图](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/easyanimatev5.1_workflow_v2v_control.jpg)

您可以使用以下视频运行演示：

[演示视频](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4)