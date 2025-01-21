# 📷 EasyAnimate | 高解像度および長時間動画生成のためのエンドツーエンドソリューション
😊 EasyAnimateは、高解像度および長時間動画を生成するためのエンドツーエンドソリューションです。トランスフォーマーベースの拡散生成器をトレーニングし、長時間動画を処理するためのVAEをトレーニングし、メタデータを前処理することができます。

😊 DITをベースに、トランスフォーマーを拡散器として使用して動画や画像を生成します。

😊 ようこそ！

[![Arxiv Page](https://img.shields.io/badge/Arxiv-Page-red)](https://arxiv.org/abs/2405.18991)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://easyanimate.github.io/)
[![Modelscope Studio](https://img.shields.io/badge/Modelscope-Studio-blue)](https://modelscope.cn/studios/PAI/EasyAnimate/summary)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/alibaba-pai/EasyAnimate)
[![Discord Page](https://img.shields.io/badge/Discord-Page-blue)](https://discord.gg/UzkpB4Bn)

English | [简体中文](./README_zh-CN.md) | 日本語

# 目次
- [目次](#目次)
- [紹介](#紹介)
- [クイックスタート](#クイックスタート)
- [ビデオ結果](#ビデオ結果)
- [使い方](#使い方)
- [モデルズー](#モデルズー)
- [TODOリスト](#todoリスト)
- [お問い合わせ](#お問い合わせ)
- [参考文献](#参考文献)
- [ライセンス](#ライセンス)

# 紹介
EasyAnimateは、トランスフォーマーアーキテクチャに基づいたパイプラインで、AI画像および動画の生成、Diffusion TransformerのベースラインモデルおよびLoraモデルのトレーニングに使用されます。事前トレーニング済みのEasyAnimateモデルから直接予測を行い、さまざまな解像度で約6秒間、8fpsの動画を生成できます（EasyAnimateV5、1〜49フレーム）。さらに、ユーザーは特定のスタイル変換のために独自のベースラインおよびLoraモデルをトレーニングできます。

異なるプラットフォームからのクイックプルアップをサポートします。詳細は[クイックスタート](#クイックスタート)を参照してください。

**新機能:**
- インセンティブ逆伝播を使用してLoraを訓練し、人間の好みに合うようにビデオを最適化します。詳細は、[ここ]（scripts/README _ train _ REVARD.md）を参照してください。EasyAnimateV 5-7 bがリリースされました。[2024.11.27]
- **v5に更新**、1024x1024までの動画生成をサポート、49フレーム、6秒、8fps、モデルスケールを12Bに拡張、MMDIT構造を組み込み、さまざまな入力を持つ制御モデルをサポート。中国語と英語のバイリンガル予測をサポート。[2024.11.08]
- **v4に更新**、1024x1024までの動画生成をサポート、144フレーム、6秒、24fps、テキスト、画像、動画からの動画生成をサポート、512から1280までの解像度を単一モデルで処理。中国語と英語のバイリンガル予測をサポート。[2024.08.15]
- **v3に更新**、960x960までの動画生成をサポート、144フレーム、6秒、24fps、テキストと画像からの動画生成をサポート。[2024.07.01]
- **ModelScope-Sora “データディレクター” クリエイティブレース** — 第三回Data-Juicerビッグモデルデータチャレンジが正式に開始されました！EasyAnimateをベースモデルとして使用し、データ処理がモデルトレーニングに与える影響を探ります。詳細は[競技ウェブサイト](https://tianchi.aliyun.com/competition/entrance/532219)をご覧ください。[2024.06.17]
- **v2に更新**、768x768までの動画生成をサポート、144フレーム、6秒、24fps。[2024.05.26]
- **コード作成！** 現在、WindowsおよびLinuxをサポート。[2024.04.12]

機能：
- [データ前処理](#data-preprocess)
- [VAEのトレーニング](#vae-train)
- [DiTのトレーニング](#dit-train)
- [動画生成](#video-gen)

私たちのUIインターフェースは次のとおりです：
![ui](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/ui_v3.jpg)

# クイックスタート
### 1. クラウド使用: AliyunDSW/Docker
#### a. AliyunDSWから
DSWには無料のGPU時間があり、ユーザーは一度申請でき、申請後3ヶ月間有効です。

Aliyunは[Freetier](https://free.aliyun.com/?product=9602825&crowd=enterprise&spm=5176.28055625.J_5831864660.1.e939154aRgha4e&scm=20140722.M_9974135.P_110.MO_1806-ID_9974135-MID_9974135-CID_30683-ST_8512-V_1)で無料のGPU時間を提供しており、取得してAliyun PAI-DSWで使用し、5分以内にEasyAnimateを開始できます！

[![DSW Notebook](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/dsw.png)](https://gallery.pai-ml.com/#/preview/deepLearning/cv/easyanimate_v5)

#### b. ComfyUIから
私たちのComfyUIは次のとおりです。詳細は[ComfyUI README](comfyui/README.md)を参照してください。
![workflow graph](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v3/comfyui_i2v.jpg)

#### c. Dockerから
Dockerを使用している場合は、マシンにグラフィックスカードドライバとCUDA環境が正しくインストールされていることを確認してください。

次のコマンドを実行します：
```
# イメージをプル
docker pull mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# イメージに入る
docker run -it -p 7860:7860 --network host --gpus all --security-opt seccomp:unconfined --shm-size 200g mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easycv/torch_cuda:easyanimate

# コードをクローン
git clone https://github.com/aigc-apps/EasyAnimate.git

# EasyAnimateのディレクトリに入る
cd EasyAnimate

# 重みをダウンロード
mkdir models/Diffusion_Transformer
mkdir models/Motion_Module
mkdir models/Personalized_Model

# EasyAnimateV5モデルをダウンロードするには、hugginfaceリンクまたはmodelscopeリンクを使用してください。
# I2Vモデル
# https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh-InP
# https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh-InP
# T2Vモデル
# https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh
# https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh
```

### 2. ローカルインストール: 環境チェック/ダウンロード/インストール
#### a. 環境チェック
次の環境でEasyAnimateの実行を確認しました：

Windowsの詳細：
- OS: Windows 10
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU： Nvidia-3060 12G

Linuxの詳細：
- OS: Ubuntu 20.04, CentOS
- python: python3.10 & python3.11
- pytorch: torch2.2.0
- CUDA: 11.8 & 12.1
- CUDNN: 8+
- GPU：Nvidia-V100 16G & Nvidia-A10 24G & Nvidia-A100 40G & Nvidia-A100 80G

ディスクに約60GBの空き容量が必要です（重みを保存するため）、確認してください！

EasyAnimateV5-12Bのビデオサイズは異なるGPUメモリにより生成できます。以下の表をご覧ください：
| GPUメモリ |384x672x72|384x672x49|576x1008x25|576x1008x49|768x1344x25|768x1344x49|
|----------|----------|----------|----------|----------|----------|----------|
| 16GB | 🧡 | 🧡 | ❌ | ❌ | ❌ | ❌ | 
| 24GB | 🧡 | 🧡 | 🧡 | 🧡 | ❌ | ❌ | 
| 40GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 
| 80GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 

EasyAnimateV5-7Bのビデオサイズは異なるGPUメモリにより生成できます。以下の表をご覧ください：
| GPUメモリ |384x672x72|384x672x49|576x1008x25|576x1008x49|768x1344x25|768x1344x49|
|----------|----------|----------|----------|----------|----------|----------|
| 16GB | 🧡 | 🧡 | ❌ | ❌ | ❌ | ❌ | 
| 24GB | ✅ | ✅ | 🧡 | 🧡 | ❌ | ❌ | 
| 40GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 
| 80GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 

✅ は"model_cpu_offload"の条件で実行可能であることを示し、🧡は"model_cpu_offload_and_qfloat8"の条件で実行可能を示し、⭕️ は"sequential_cpu_offload"の条件では実行可能であることを示しています。❌は実行できないことを示します。sequential_cpu_offloadにより実行する場合は遅くなります。

一部のGPU（例：2080ti、V100）はtorch.bfloat16をサポートしていないため、app.pyおよびpredictファイル内のweight_dtypeをtorch.float16に変更する必要があります。

EasyAnimateV5-12Bは異なるGPUで25ステップ生成する時間は次の通りです：
| GPU |384x672x72|384x672x49|576x1008x25|576x1008x49|768x1344x25|768x1344x49|
|----------|----------|----------|----------|----------|----------|----------|
| A10 24GB |約120秒 (4.8s/it)|約240秒 (9.6s/it)|約320秒 (12.7s/it)|約750秒 (29.8s/it)| ❌ | ❌ |
| A100 80GB |約45秒 (1.75s/it)|約90秒 (3.7s/it)|約120秒 (4.7s/it)|約300秒 (11.4s/it)|約265秒 (10.6s/it)| 約710秒 (28.3s/it)|

<details>
  <summary>(廃止予定) EasyAnimateV3:</summary>
EasyAnimateV3のビデオサイズは異なるGPUメモリにより生成できます。以下の表をご覧ください：
| GPUメモリ | 384x672x72 | 384x672x144 | 576x1008x72 | 576x1008x144 | 720x1280x72 | 720x1280x144 |
|----------|----------|----------|----------|----------|----------|----------|
| 12GB | ⭕️ | ⭕️ | ⭕️ | ⭕️ | ❌ | ❌ |
| 16GB | ✅ | ✅ | ⭕️ | ⭕️ | ⭕️ | ❌ |
| 24GB | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| 40GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 80GB | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

(⭕️) はlow_gpu_memory_mode=Trueの条件で実行可能であるが、速度が遅くなることを示しています。また、❌は実行できないことを示します。
</details>

#### b. 重み
[重み](#model-zoo)を指定されたパスに配置することをお勧めします：

EasyAnimateV5:
```
📦 models/
├── 📂 Diffusion_Transformer/
│   ├── 📂 EasyAnimateV5-12b-zh-InP/
│   └── 📂 EasyAnimateV5-12b-zh/
├── 📂 Personalized_Model/
│   └── あなたのトレーニング済みのトランスフォーマーモデル / あなたのトレーニング済みのLoraモデル（UIロード用）
```

# ビデオ結果

### Image to Video with EasyAnimateV5.1-12b-zh-InP
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/74a23109-f555-4026-a3d8-1ac27bb3884c" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/ab5aab27-fbd7-4f55-add9-29644125bde7" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/238043c2-cdbd-4288-9857-a273d96f021f" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/48881a0e-5513-4482-ae49-13a0ad7a2557" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>


<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3e7aba7f-6232-4f39-80a8-6cfae968f38c" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/986d9f77-8dc3-45fa-bc9d-8b26023fffbc" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/7f62795a-2b3b-4c14-aeb1-1230cb818067" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b581df84-ade1-4605-a7a8-fd735ce3e222
          " width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/eab1db91-1082-4de2-bb0a-d97fd25ceea1" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/3fda0e96-c1a8-4186-9c4c-043e11420f05" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/4b53145d-7e98-493a-83c9-4ea4f5b58289" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/75f7935f-17a8-4e20-b24c-b61479cf07fc" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### Text to Video with EasyAnimateV5.1-12b-zh
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/8818dae8-e329-4b08-94fa-00d923f38fd2" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/d3e483c3-c710-47d2-9fac-89f732f2260a" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/4dfa2067-d5d4-4741-a52c-97483de1050d" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fb44c2db-82c6-427e-9297-97dcce9a4948" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/dc6b8eaf-f21b-4576-a139-0e10438f20e4" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b3f8fd5b-c5c8-44ee-9b27-49105a08fbff" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/a68ed61b-eed3-41d2-b208-5f039bf2788e" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/4e33f512-0126-4412-9ae8-236ff08bcd21" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### Control Video with EasyAnimateV5.1-12b-zh-Control

Trajectory Control:
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/bf3b8970-ca7b-447f-8301-72dfe028055b" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/63a7057b-573e-4f73-9d7b-8f8001245af4" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/090ac2f3-1a76-45cf-abe5-4e326113389b" width="100%" controls autoplay loop></video>
     </td>
  <tr>
</table>

Generic Control Video (Canny, Pose, Depth, etc.):
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/53002ce2-dd18-4d4f-8135-b6f68364cabd" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/fce43c0b-81fa-4ab2-9ca7-78d786f520e6" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/b208b92c-5add-4ece-a200-3dbbe47b93c3" width="100%" controls autoplay loop></video>
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/3aec95d5-d240-49fb-a9e9-914446c7a4cf" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/60fa063b-5c1f-485f-b663-09bd6669de3f" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/4adde728-8397-42f3-8a2a-23f7b39e9a1e" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

### Camera Control with EasyAnimateV5.1-12b-zh-Control-Camera

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          Pan Up
      </td>
      <td>
          Pan Left
      </td>
       <td>
          Pan Right
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/a88f81da-e263-4038-a5b3-77b26f79719e" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/e346c59d-7bca-4253-97fb-8cbabc484afb" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/4de470d4-47b7-46e3-82d3-b714a2f6aef6" width="100%" controls autoplay loop></video>
     </td>
  <tr>
      <td>
          Pan Down
      </td>
      <td>
          Pan Up + Pan Left
      </td>
       <td>
          Pan Up + Pan Right
     </td>
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/7a3fecc2-d41a-4de3-86cd-5e19aea34a0d" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/cb281259-28b6-448e-a76f-643c3465672e" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/44faf5b6-d83c-4646-9436-971b2b9c7216" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

# 使い方

<h3 id="video-gen">1. 推論 </h3>

#### a、メモリ節約策
EasyAnimateV5およびV5.1のパラメータが非常に大きいため、消費者向けグラフィックスカードに適応させるためにメモリの節約策を考慮する必要があります。各予測ファイルにはGPU_memory_modeを提供しており、model_cpu_offload、model_cpu_offload_and_qfloat8、sequential_cpu_offloadから選択することができます。

- model_cpu_offloadは、使用後にモデル全体がCPUに移動することを示し、メモリの一部を節約できます。
- model_cpu_offload_and_qfloat8は、使用後にモデル全体がCPUに移動し、トランスフォーマーモデルをfloat8に量子化することを示し、さらに多くのメモリを節約できます。
- sequential_cpu_offloadは、使用後に各レイヤーが順次CPUに移動することを示し、速度は遅くなりますが、大量のメモリを節約できます。

qfloat8はモデルの性能を低下させますが、さらに多くのメモリを節約できます。メモリが十分にある場合は、model_cpu_offloadを使用することをお勧めします。

#### b、ComfyUIを使用する
詳細は[ComfyUI README](comfyui/README.md)をご覧ください。

#### c、pythonファイルを実行する
- ステップ1：対応する[重み](#model-zoo)をダウンロードし、modelsフォルダに入れます。
- ステップ2：異なる重みと予測目標に応じて異なるファイルを使用して予測を行います。
  - テキストからビデオの生成：
    - predict_t2v.pyファイルでprompt、neg_prompt、guidance_scale、seedを変更します。
    - 次にpredict_t2v.pyファイルを実行し、生成結果を待ちます。結果はsamples/easyanimate-videosフォルダに保存されます。
  - 画像からビデオの生成：
    - predict_i2v.pyファイルでvalidation_image_start、validation_image_end、prompt、neg_prompt、guidance_scale、seedを変更します。
    - validation_image_startはビデオの開始画像、validation_image_endはビデオの終了画像です。
    - 次にpredict_i2v.pyファイルを実行し、生成結果を待ちます。結果はsamples/easyanimate-videos_i2vフォルダに保存されます。
  - ビデオからビデオの生成：
    - predict_v2v.pyファイルでvalidation_video、validation_image_end、prompt、neg_prompt、guidance_scale、seedを変更します。
    - validation_videoはビデオの参照ビデオです。以下のビデオを使用してデモを実行できます：[デモビデオ](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/play_guitar.mp4)
    - 次にpredict_v2v.pyファイルを実行し、生成結果を待ちます。結果はsamples/easyanimate-videos_v2vフォルダに保存されます。
  - 通常のコントロールビデオ生成（Canny、Pose、Depthなど）：
    - predict_v2v_control.pyファイルでcontrol_video、validation_image_end、prompt、neg_prompt、guidance_scale、seedを変更します。
    - control_videoはCanny、Pose、Depthなどのフィルタを適用した後のビデオです。以下のビデオを使用してデモを実行できます：[デモビデオ](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1.1/pose.mp4)
    - 次にpredict_v2v_control.pyファイルを実行し、生成結果を待ちます。結果はsamples/easyanimate-videos_v2v_controlフォルダに保存されます。
  - トラジェクトリーコントロールビデオ：
    - predict_v2v_control.pyファイルでcontrol_video、ref_image、validation_image_end、prompt、neg_prompt、guidance_scale、seedを変更します。
    - control_videoはトラジェクトリーコントロールビデオのコントロールビデオ、ref_imageは参照の初期フレーム画像です。以下の画像とコントロールビデオを使用してデモを実行できます：[デモ画像](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/dog.png)、[デモビデオ](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/trajectory_demo.mp4)
    - 次にpredict_v2v_control.pyファイルを実行し、生成結果を待ちます。結果はsamples/easyanimate-videos_v2v_controlフォルダに保存されます。
    - 交互利用にComfyUIの使用を推奨します。
  - カメラコントロールビデオ：
    - predict_v2v_control.pyファイルでcontrol_video、ref_image、validation_image_end、prompt、neg_prompt、guidance_scale、seedを変更します。
    - control_camera_txtはカメラコントロールビデオのコントロールファイル、ref_imageは参照の初期フレーム画像です。以下の画像とコントロールビデオを使用してデモを実行できます：[デモ画像](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/cogvideox_fun/asset/v1/firework.png)、[デモファイル（CameraCtrlから）](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v5.1/0a3b5fb184936a83.txt)
    - 次にpredict_v2v_control.pyファイルを実行し、生成結果を待ちます。結果はsamples/easyanimate-videos_v2v_controlフォルダに保存されます。
    - 交互利用にComfyUIの使用を推奨します。
- ステップ3：他のトレーニング済みバックボーンとLoraを組み合わせたい場合、predict_t2v.pyでpredict_t2v.pyとlora_pathを適宜変更してください。

#### d、UIインターフェイスを使用する

webuiはテキストからビデオ、画像からビデオ、ビデオからビデオ、および通常のコントロールビデオ（Canny、Pose、Depthなど）の生成をサポートしています。

- ステップ1：対応する[重み](#model-zoo)をダウンロードし、modelsフォルダに入れます。
- ステップ2：app.pyファイルを実行し、gradioページに入ります。
- ステップ3：ページで生成モデルを選択し、prompt、neg_prompt、guidance_scale、seedなどを入力して生成をクリックし、生成結果を待ちます。結果はsampleフォルダに保存されます。

### 2. モデルトレーニング
完全なEasyAnimateトレーニングパイプラインには、データ前処理、Video VAEトレーニング、およびVideo DiTトレーニングが含まれる必要があります。これらの中で、Video VAEトレーニングはオプションです。すでにトレーニング済みのVideo VAEを提供しているためです。

<h4 id="data-preprocess">a. データ前処理</h4>

画像データを使用してLoraモデルをトレーニングする簡単なデモを提供しています。詳細は[wiki](https://github.com/aigc-apps/EasyAnimate/wiki/Training-Lora)を参照してください。

長時間動画のセグメンテーション、クリーニング、および説明のための完全なデータ前処理リンクは、ビデオキャプションセクションの[README](./easyanimate/video_caption/README.md)を参照してください。

テキストから画像および動画生成モデルをトレーニングする場合は、データセットを次の形式で配置する必要があります。

```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

json_of_internal_datasets.jsonは標準のJSONファイルです。json内のfile_pathは相対パスとして設定できます。以下のように：
```json
[
    {
      "file_path": "train/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

パスを絶対パスとして設定することもできます：
```json
[
    {
      "file_path": "/mnt/data/videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "/mnt/data/train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    .....
]
```

<h4 id="vae-train">b. Video VAEトレーニング（オプション）</h4>

Video VAEトレーニングはオプションです。すでにトレーニング済みのVideo VAEを提供しているためです。
Video VAEをトレーニングする場合は、ビデオVAEセクションの[README](easyanimate/vae/README.md)を参照してください。

<h4 id="dit-train">c. Video DiTトレーニング </h4>

データ前処理時にデータ形式が相対パスの場合、```scripts/train.sh```を次のように設定します。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"
```

データ前処理時にデータ形式が絶対パスの場合、```scripts/train.sh```を次のように設定します。
```
export DATASET_NAME=""
export DATASET_META_NAME="/mnt/data/json_of_internal_datasets.json"
```

次に、scripts/train.shを実行します。
```sh
sh scripts/train.sh
```

一部のパラメータの設定の詳細については、[Readme Train](scripts/README_TRAIN.md)および[Readme Lora](scripts/README_TRAIN_LORA.md)を参照してください。

<details>
  <summary>(Obsolete) EasyAnimateV1:</summary>
  EasyAnimateV1をトレーニングする場合は、gitブランチv1に切り替えてください。
</details>

# モデルズー

12B:
| 名前 | タイプ | ストレージスペース | Hugging Face | モデルスコープ | 説明 |
|--|--|--|--|--|--|
| EasyAnimateV5.1-12b-zh-InP | EasyAnimateV5.1 | 39 GB | [🤗リンク](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-InP) | [😄リンク](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-InP) | 公式の画像からビデオへの変換用の重み。支持多解像度（512、768、1024）的ビデオ予測、49フレームで毎秒8フレームの訓練、多言語予測をサポート |
| EasyAnimateV5.1-12b-zh-Control | EasyAnimateV5.1 | 39 GB | [🤗リンク](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control) | [😄リンク](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-Control) | 公式のビデオ制御用の重み。Canny、Depth、Pose、MLSD、および軌道制御などのさまざまな制御条件をサポートします。支持多解像度（512、768、1024）的ビデオ予測、49フレームで毎秒8フレームの訓練、多言語予測をサポート |
| EasyAnimateV5.1-12b-zh-Control-Camera | EasyAnimateV5.1 | 39 GB | [🤗リンク](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh-Control-Camera) | [😄リンク](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh-Control-Camera) | 公式のビデオカメラ制御用の重み。カメラの動きの軌跡を入力することで方向生成を制御します。支持多解像度（512、768、1024）的ビデオ予測、49フレームで毎秒8フレームの訓練、多言語予測をサポート |
| EasyAnimateV5.1-12b-zh | EasyAnimateV5.1 | 39 GB | [🤗リンク](https://huggingface.co/alibaba-pai/EasyAnimateV5.1-12b-zh) | [😄リンク](https://modelscope.cn/models/PAI/EasyAnimateV5.1-12b-zh) | 公式のテキストからビデオへの変換用の重み。支持多解像度（512、768、1024）的ビデオ予測、49フレームで毎秒8フレームの訓練、多言語予測をサポート |

<details>
  <summary>(Obsolete) EasyAnimateV5:</summary>

7B:
| 名前 | 種類 | ストレージスペース | Hugging Face | Model Scope | 説明 |
|--|--|--|--|--|--|
| EasyAnimateV5-7b-zh-InP | EasyAnimateV5 | 22 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-7b-zh-InP) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-7b-zh-InP) | 公式の画像から動画への重み。複数の解像度（512、768、1024）での動画予測をサポートし、49フレーム、毎秒8フレームでトレーニングされ、中国語と英語のバイリンガル予測をサポートします。 |
| EasyAnimateV5-7b-zh | EasyAnimateV5 | 22 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-7b-zh) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-7b-zh) | 公式のテキストから動画への重み。複数の解像度（512、768、1024）での動画予測をサポートし、49フレーム、毎秒8フレームでトレーニングされ、中国語と英語のバイリンガル予測をサポートします。 |
| EasyAnimateV5-Reward-LoRAs | EasyAnimateV5 | - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-Reward-LoRAs) | 公式インバース伝播技術モデルによるEasyAnimateV 5-12 b生成ビデオの最適化によるヒト選好の最適化｜

12B:
| 名前 | 種類 | ストレージスペース | Hugging Face | Model Scope | 説明 |
|--|--|--|--|--|--|
| EasyAnimateV5-12b-zh-InP | EasyAnimateV5 | 34 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh-InP) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh-InP) | 公式の画像から動画への重み。複数の解像度（512、768、1024）での動画予測をサポートし、49フレーム、毎秒8フレームでトレーニングされ、中国語と英語のバイリンガル予測をサポートします。 |
| EasyAnimateV5-12b-zh-Control | EasyAnimateV5 | 34 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh-Control) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh-Control) | 公式の動画制御重み。Canny、Depth、Pose、MLSDなどのさまざまな制御条件をサポートします。複数の解像度（512、768、1024）での動画予測をサポートし、49フレーム、毎秒8フレームでトレーニングされ、中国語と英語のバイリンガル予測をサポートします。 |
| EasyAnimateV5-12b-zh | EasyAnimateV5 | 34 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-12b-zh) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-12b-zh) | 公式のテキストから動画への重み。複数の解像度（512、768、1024）での動画予測をサポートし、49フレーム、毎秒8フレームでトレーニングされ、中国語と英語のバイリンガル予測をサポートします。 |
| EasyAnimateV5-Reward-LoRAs | EasyAnimateV5 | - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV5-Reward-LoRAs) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV5-Reward-LoRAs) | 公式インバース伝播技術モデルによるEasyAnimateV 5-12 b生成ビデオの最適化によるヒト選好の最適化｜
</details>

<details>
  <summary>(Obsolete) EasyAnimateV4:</summary>

| 名前 | 種類 | ストレージスペース | Hugging Face | Model Scope | 説明 |
|--|--|--|--|--|--|
| EasyAnimateV4-XL-2-InP | EasyAnimateV4 | 解凍前: 8.9 GB / 解凍後: 14.0 GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV4-XL-2-InP)| [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV4-XL-2-InP) | 公式のグラフ生成動画モデル。複数の解像度（512、768、1024、1280）での動画予測をサポートし、144フレーム、毎秒24フレームでトレーニングされています。 |
</details>

<details>
  <summary>(Obsolete) EasyAnimateV3:</summary>

| 名前 | 種類 | ストレージスペース | Hugging Face | Model Scope | 説明 |
|--|--|--|--|--|--|
| EasyAnimateV3-XL-2-InP-512x512 | EasyAnimateV3 | 18.2GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-512x512)| [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-512x512) | EasyAnimateV3公式の512x512テキストおよび画像から動画への重み。144フレーム、毎秒24フレームでトレーニングされています。 |
| EasyAnimateV3-XL-2-InP-768x768 | EasyAnimateV3 | 18.2GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-768x768) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-768x768) | EasyAnimateV3公式の768x768テキストおよび画像から動画への重み。144フレーム、毎秒24フレームでトレーニングされています。 |
| EasyAnimateV3-XL-2-InP-960x960 | EasyAnimateV3 | 18.2GB | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV3-XL-2-InP-960x960) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV3-XL-2-InP-960x960) | EasyAnimateV3公式の960x960テキストおよび画像から動画への重み。144フレーム、毎秒24フレームでトレーニングされています。 |
</details>

<details>
  <summary>(Obsolete) EasyAnimateV2:</summary>

| 名前 | 種類 | ストレージスペース | URL | Hugging Face | Model Scope | 説明 |
|--|--|--|--|--|--|--|
| EasyAnimateV2-XL-2-512x512 | EasyAnimateV2 | 16.2GB |  - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-512x512)| [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV2-XL-2-512x512) | EasyAnimateV2公式の512x512解像度の重み。144フレーム、毎秒24フレームでトレーニングされています。 |
| EasyAnimateV2-XL-2-768x768 | EasyAnimateV2 | 16.2GB | - | [🤗Link](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-768x768) | [😄Link](https://modelscope.cn/models/PAI/EasyAnimateV2-XL-2-768x768) | EasyAnimateV2公式の768x768解像度の重み。144フレーム、毎秒24フレームでトレーニングされています。 |
| easyanimatev2_minimalism_lora.safetensors | Lora of Pixart | 485.1MB | [ダウンロード](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimatev2_minimalism_lora.safetensors) | - | - | 特定のタイプの画像でトレーニングされたLora。画像は[URL](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/v2/Minimalism.zip)からダウンロードできます。 |
</details>

<details>
  <summary>(Obsolete) EasyAnimateV1:</summary>

### 1、モーション重み
| 名前 | 種類 | ストレージスペース | URL | 説明 |
|--|--|--|--|--| 
| easyanimate_v1_mm.safetensors | モーションモジュール | 4.1GB | [ダウンロード](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Motion_Module/easyanimate_v1_mm.safetensors) | 80フレーム、毎秒12フレームでトレーニングされています。 |

### 2、その他の重み
| 名前 | 種類 | ストレージスペース | URL | 説明 |
|--|--|--|--|--| 
| PixArt-XL-2-512x512.tar | Pixart | 11.4GB | [ダウンロード](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Diffusion_Transformer/PixArt-XL-2-512x512.tar)| Pixart-Alpha公式の重み。 |
| easyanimate_portrait.safetensors | Pixartのチェックポイント | 2.3GB | [ダウンロード](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimate_portrait.safetensors) | 内部のポートレートデータセットでトレーニングされています。 |
| easyanimate_portrait_lora.safetensors | PixartのLora | 654.0MB | [ダウンロード](https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/Personalized_Model/easyanimate_portrait_lora.safetensors)| 内部のポートレートデータセットでトレーニングされています。 |
</details>

# TODOリスト
- より大きなパラメータを持つモデルをサポートします。

# お問い合わせ
1. Dingdingを使用してグループ77450006752を検索するか、スキャンして参加します。
2. WeChatグループに参加するには画像をスキャンするか、期限切れの場合はこの学生を友達として追加して招待します。

<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/group/dd.png" alt="ding group" width="30%"/>
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/group/wechat.jpg" alt="Wechat group" width="30%"/>
<img src="https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/asset/group/person.jpg" alt="Person" width="30%"/>


# 参考文献
- CogVideo: https://github.com/THUDM/CogVideo/
- Flux: https://github.com/black-forest-labs/flux
- magvit: https://github.com/google-research/magvit
- PixArt: https://github.com/PixArt-alpha/PixArt-alpha
- Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Open-Sora: https://github.com/hpcaitech/Open-Sora
- Animatediff: https://github.com/guoyww/AnimateDiff
- HunYuan DiT: https://github.com/tencent/HunyuanDiT
- ComfyUI-KJNodes: https://github.com/kijai/ComfyUI-KJNodes
- ComfyUI-EasyAnimateWrapper: https://github.com/kijai/ComfyUI-EasyAnimateWrapper
- ComfyUI-CameraCtrl-Wrapper: https://github.com/chaojie/ComfyUI-CameraCtrl-Wrapper
- CameraCtrl: https://github.com/hehao13/CameraCtrl
- DragAnything: https://github.com/showlab/DragAnything

# ライセンス
このプロジェクトは[Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE)の下でライセンスされています。
