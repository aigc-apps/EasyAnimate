import csv
import io
import json
import math
import os
import random
from threading import Thread

import albumentations
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset


class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]

class ImageVideoDataset(Dataset):
    def __init__(
            self,
            ann_path, data_root=None,
            video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
            image_sample_size=512,
            # For Random Crop
            min_crop_f=0.9, max_crop_f=1,
            video_repeat=0,
            enable_bucket=False
        ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        self.dataset = []
        for data in dataset:
            if data.get('data_type', 'image') != 'video' or data.get('type', 'image') != 'video':
                self.dataset.append(data)
        if video_repeat > 0:
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('data_type', 'image') == 'video' or data.get('type', 'image') == 'video':
                        self.dataset.append(data)
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        
        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size      = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_rescaler         = albumentations.SmallestMaxSize(max_size=min(self.video_sample_size), interpolation=cv2.INTER_AREA)

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
    
    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        
        if data_info.get('data_type', 'image')=='video' or data_info.get('type', 'image')=='video':
            video_path, text = data_info['file_path'], data_info['text']
            # Get abs path of video
            if self.data_root is not None:
                video_path = os.path.join(self.data_root, video_path)

            # Get video information firstly
            video_reader = VideoReader(video_path, num_threads=2)
            h, w, c = video_reader[0].shape
            del video_reader
            
            # Resize to bigger firstly
            t_h = int(self.video_sample_size[0] * 1.25 * h / min(h, w))
            t_w = int(self.video_sample_size[0] * 1.25 * w / min(h, w))

            # Get video pixels
            video_reader    = VideoReader(video_path, width=t_w, height=t_h, num_threads=2)
            video_length    = len(video_reader)
            clip_length     = min(video_length, (self.video_sample_n_frames - 1) * self.video_sample_stride + 1)
            start_idx       = random.randint(0, video_length - clip_length)
            batch_index     = np.linspace(start_idx, start_idx + clip_length - 1, self.video_sample_n_frames, dtype=int)
            imgs            = video_reader.get_batch(batch_index).asnumpy()
            del video_reader
            if imgs.shape[0] != self.video_sample_n_frames:
                raise ValueError('Video data Sampler Error')

            # Crop center of above videos 
            min_side_len    = min(imgs[0].shape[:2])
            crop_side_len   = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
            crop_side_len   = int(crop_side_len)
            self.cropper    = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
            imgs            = np.transpose(imgs, (1, 2, 3, 0))
            imgs            = self.cropper(image=imgs)["image"]
            imgs            = np.transpose(imgs, (3, 0, 1, 2))
            out_imgs        = []

            # Resize to video_sample_size
            for img in imgs:
                img = self.video_rescaler(image=img)["image"]
                out_imgs.append(img[None, :, :, :])
            imgs = np.concatenate(out_imgs).transpose(0, 3, 1, 2)

            # Normalize to -1～1
            imgs = ((imgs - 127.5) / 127.5).astype(np.float32)
            if imgs.shape[0] != self.video_sample_n_frames:
                raise ValueError('video data sampler error')
            
            # Random use no text generation
            if random.random() < 0.1:
                text = ''
            return torch.from_numpy(imgs), text, 'video'
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            image = self.image_transforms(image).unsqueeze(0)
            if random.random()<0.1:
                text = ''
            return image, text, 'video'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            sample = {}
            def get_data(data_idx):
                pixel_values, name, data_type = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
            try:
                t = Thread(target=get_data, args=(idx, ))
                t.start()
                t.join(5)
                if len(sample)>0:
                    break
            except Exception as e:
                print(self.dataset[idx])
                idx = idx - 1
        return sample

if __name__ == "__main__":
    dataset = ImageVideoDataset(
        ann_path="test.json"
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))