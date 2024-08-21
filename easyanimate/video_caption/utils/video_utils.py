import gc
import random
from contextlib import contextmanager
from typing import List, Tuple, Optional

import numpy as np
from decord import VideoReader
from PIL import Image


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
    video_path: str,
    sample_method: str = "mid",
    num_sampled_frames: int = -1,
    sample_stride: int = -1,
    **kwargs
) -> Optional[Tuple[List[int], List[Image.Image]]]:
    with video_reader(video_path, num_threads=2, **kwargs) as vr:
        if sample_method == "mid":
            sampled_frame_idx_list = [len(vr) // 2]
        elif sample_method == "uniform":
            sampled_frame_idx_list = np.linspace(0, len(vr), num_sampled_frames, endpoint=False, dtype=int)
        elif sample_method == "random":
            clip_length = min(len(vr), (num_sampled_frames - 1) * sample_stride + 1)
            start_idx = random.randint(0, len(vr) - clip_length)
            sampled_frame_idx_list = np.linspace(start_idx, start_idx + clip_length - 1, num_sampled_frames, dtype=int)
        else:
            raise ValueError(f"The sample_method {sample_method} must be mid, uniform or random.")
        sampled_frame_list = vr.get_batch(sampled_frame_idx_list).asnumpy()
        sampled_frame_list = [Image.fromarray(frame) for frame in sampled_frame_list]

        return list(sampled_frame_idx_list), sampled_frame_list
