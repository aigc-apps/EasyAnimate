## EN、Training video generation model
### i、Base on webvid dataset
If using the webvid dataset for training, you need to download the webvid dataset firstly.

You need to arrange the webvid dataset in this format.

```
📦 project/
├── 📂 datasets/
│   ├── 📂 webvid/
│       ├── 📂 videos/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.mp4
│       │   └── 📄 .....
│       └── 📄 csv_of_webvid.csv
```

Then，set scripts/train_t2v.sh.
```
export DATASET_NAME="datasets/webvid/videos/"
export DATASET_META_NAME="datasets/webvid/csv_of_webvid.csv"

...

train_data_format="webvid"
```

Then, we run scripts/train_t2v.sh.
```sh
sh scripts/train_t2v.sh
```

### ii、Base on internal dataset
If using the internal dataset for training, you need to format the dataset firstly.

You need to arrange the dataset in this format.

```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 videos/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.mp4
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

The json_of_internal_datasets.json is a standard JSON file, as shown in below:
```json
[
    {
      "file_path": "videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "videos/00000002.mp4",
      "text": "A notepad with a drawing of a woman on it.",
      "type": "video"
    }
    .....
]
```
The file_path in the json needs to be set as relative path.

Then, set scripts/train_t2v.sh.
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"

...

train_data_format="normal"
```

Then, we run scripts/train_t2v.sh.
```sh
sh scripts/train_t2v.sh
```

## CN、训练视频生成模型
### i、基于webvid数据集
如果使用webvid数据集进行训练，则需要首先下载webvid的数据集。

您需要以这种格式排列webvid数据集。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 webvid/
│       ├── 📂 videos/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.mp4
│       │   └── 📄 .....
│       └── 📄 csv_of_webvid.csv
```

然后，进入scripts/train_t2v.sh进行设置。
```
export DATASET_NAME="datasets/webvid/videos/"
export DATASET_META_NAME="datasets/webvid/csv_of_webvid.csv"

...

train_data_format="webvid"
```

最后运行scripts/train_t2v.sh。
```sh
sh scripts/train_t2v.sh
```

### ii、基于自建数据集
如果使用内部数据集进行训练，则需要首先格式化数据集。

您需要以这种格式排列数据集。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 videos/
│       │   ├── 📄 00000001.mp4
│       │   ├── 📄 00000002.mp4
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

json_of_internal_datasets.json是一个标准的json文件，如下所示：
```json
[
    {
      "file_path": "videos/00000001.mp4",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "video"
    },
    {
      "file_path": "videos/00000002.mp4",
      "text": "A notepad with a drawing of a woman on it.",
      "type": "video"
    }
    .....
]
```
json中的file_path需要设置为相对路径。

然后，进入scripts/train_t2v.sh进行设置。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"

...

train_data_format="normal"
```

最后运行scripts/train_t2v.sh。
```sh
sh scripts/train_t2v.sh
```