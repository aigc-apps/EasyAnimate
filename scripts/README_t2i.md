## EN、Training text to image model
### i、Base on diffusers format
The format of dataset can be set as diffuser format.
If using the diffusers format dataset for training.

```
📦 project/
├── 📂 datasets/
│   ├── 📂 diffusers_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.jpg
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 metadata.jsonl
```

Then, set scripts/train_t2i.sh.
```
export DATASET_NAME="datasets/diffusers_datasets/"

...

train_data_format="diffusers"
```

Then, we run scripts/train_t2i.sh.
```sh
sh scripts/train_t2i.sh
```
### ii、Base on internal dataset
If using the internal dataset for training, you need to format the dataset firstly.

You need to arrange the dataset in this format.

```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.jpg
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

The json_of_internal_datasets.json is a standard JSON file, as shown in below:
```json
[
    {
      "file_path": "train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A notepad with a drawing of a woman on it.",
      "type": "image"
    }
    .....
]
```
The file_path in the json needs to be set as relative path.

Then, set scripts/train_t2i.sh.
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"

...

train_data_format="normal"
```

Then, we run scripts/train_t2i.sh.
```sh
sh scripts/train_t2i.sh
```

## CN、训练基础文生图模型
### i、基于diffusers格式
数据集的格式可以设置为diffusers格式。

```
📦 project/
├── 📂 datasets/
│   ├── 📂 diffusers_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.jpg
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 metadata.jsonl
```

然后，进入scripts/train_t2i.sh进行设置。
```
export DATASET_NAME="datasets/diffusers_datasets/"

...

train_data_format="diffusers"
```

最后运行scripts/train_t2i.sh。
```sh
sh scripts/train_t2i.sh
```
### ii、基于自建数据集
如果使用自建数据集进行训练，则需要首先格式化数据集。

您需要以这种格式排列数据集。
```
📦 project/
├── 📂 datasets/
│   ├── 📂 internal_datasets/
│       ├── 📂 train/
│       │   ├── 📄 00000001.jpg
│       │   ├── 📄 00000002.jpg
│       │   └── 📄 .....
│       └── 📄 json_of_internal_datasets.json
```

json_of_internal_datasets.json是一个标准的json文件，如下所示：
```json
[
    {
      "file_path": "train/00000001.jpg",
      "text": "A group of young men in suits and sunglasses are walking down a city street.",
      "type": "image"
    },
    {
      "file_path": "train/00000002.jpg",
      "text": "A notepad with a drawing of a woman on it.",
      "type": "image"
    }
    .....
]
```
json中的file_path需要设置为相对路径。

然后，进入scripts/train_t2i.sh进行设置。
```
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/json_of_internal_datasets.json"

...

train_data_format="normal"
```

最后运行scripts/train_t2i.sh。
```sh
sh scripts/train_t2i.sh
```