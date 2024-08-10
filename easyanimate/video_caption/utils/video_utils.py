import gc
import os
import random
import urllib.request as request
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from decord import VideoReader
from PIL import Image

ALL_VIDEO_EXT = set([".mp4", ".webm", ".mkv", ".avi", ".flv", ".mov"])


def get_video_path_list(
    video_folder: Optional[str]=None,
    video_metadata_path: Optional[str]=None,
    video_path_column: Optional[str]=None
) -> List[str]:
    """Get all video (absolute) path list from the video folder or the video metadata file.

    Args:
        video_folder (str): The absolute path of the folder (including sub-folders) containing all the required video files.
        video_metadata_path (str): The absolute path of the video metadata file containing video path list.
        video_path_column (str): The column/key for the corresponding video path in the video metadata file (csv/jsonl).
    """
    if video_folder is None and video_metadata_path is None:
        raise ValueError("Either the video_input or the video_metadata_path should be specified.")
    if video_metadata_path is not None:
        if video_metadata_path.endswith(".csv"):
            if video_path_column is None:
                raise ValueError("The video_path_column can not be None if provided a csv file.")
            metadata_df = pd.read_csv(video_metadata_path)
            video_path_list = metadata_df[video_path_column].tolist()
        elif video_metadata_path.endswith(".jsonl"):
            if video_path_column is None:
                raise ValueError("The video_path_column can not be None if provided a jsonl file.")
            metadata_df = pd.read_json(video_metadata_path, lines=True)
            video_path_list = metadata_df[video_path_column].tolist()
        elif video_metadata_path.endswith(".txt"):
            with open(video_metadata_path, "r", encoding="utf-8") as f:
                video_path_list = [line.strip() for line in f]
        else:
            raise ValueError("The video_metadata_path must end with `.csv`, `.jsonl` or `.txt`.")
        if video_folder is not None:
            video_path_list = [os.path.join(video_folder, video_path) for video_path in video_path_list]
        return video_path_list

    if os.path.isfile(video_folder):
        video_path_list = []
        if video_folder.endswith("mp4"):
            video_path_list.append(video_folder)
        elif video_folder.endswith("txt"):
            with open(video_folder, "r") as file:
                video_path_list += [line.strip() for line in file.readlines()]
        return video_path_list

    elif video_folder is not None:
        video_path_list = []
        for ext in ALL_VIDEO_EXT:
            video_path_list.extend(Path(video_folder).rglob(f"*{ext}"))
        video_path_list = [str(video_path) for video_path in video_path_list]
        return video_path_list


@contextmanager
def video_reader(*args, **kwargs):
    """A context manager to solve the memory leak of decord.
    """
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


def extract_frames(
    video_path: str, sample_method: str = "mid", num_sampled_frames: int = -1, sample_stride: int = -1
) -> Optional[Tuple[List[int], List[Image.Image]]]:
    with video_reader(video_path, num_threads=2) as vr:
        if sample_method == "mid":
            sampled_frame_idx_list = [len(vr) // 2]
        elif sample_method == "uniform":
            sampled_frame_idx_list = np.linspace(0, len(vr), num_sampled_frames, endpoint=False, dtype=int)
        elif sample_method == "random":
            clip_length = min(len(vr), (num_sampled_frames - 1) * sample_stride + 1)
            start_idx = random.randint(0, len(vr) - clip_length)
            sampled_frame_idx_list = np.linspace(start_idx, start_idx + clip_length - 1, num_sampled_frames, dtype=int)
        else:
            raise ValueError("The sample_method must be mid, uniform or random.")
        sampled_frame_list = vr.get_batch(sampled_frame_idx_list).asnumpy()
        sampled_frame_list = [Image.fromarray(frame) for frame in sampled_frame_list]

        return list(sampled_frame_idx_list), sampled_frame_list


def download_video(
    video_url: str, 
    save_path: str) -> bool:
    try:
        request.urlretrieve(video_url, save_path)
        return os.path.isfile(save_path)
    except Exception as e:
        print(e, video_url)
        return False