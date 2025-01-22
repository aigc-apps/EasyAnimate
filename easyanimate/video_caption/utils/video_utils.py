import gc
import random
import shutil
import subprocess
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
from decord import VideoReader
from PIL import Image


ALL_FRAME_SAMPLE_METHODS = [
    "mid", "uniform", "random", "stride", "first", "last", "keyframe", "keyframe+first", "keyframe+last"
]


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


def get_keyframe_index(video_path):
    """Extract the frame index list of I-frames. In general, the first frame in a video should be the I-frame.
    The extracted frame index is more accurate than the pts_time * avg_fps.
    """
    assert shutil.which("ffprobe") is not None, f"Please install ffprobe and make sure it is in the system path."
    
    command = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    keyframe_index_list = []
    frame_index = 0
    for line in result.stdout.split("\n"):
        line = line.strip(",")
        pict_type = line.strip()
        if pict_type == "I":
            keyframe_index_list.append(frame_index)
        if pict_type == "I" or pict_type == "B" or pict_type == "P":
            frame_index += 1

    return keyframe_index_list, frame_index

def extract_frames(
    video_path: str,
    sample_method: str = "mid",
    num_sampled_frames: int = 1,
    sample_stride: Optional[int] = None,
    **kwargs
) -> Optional[Tuple[List[int], List[Image.Image]]]:
    if num_sampled_frames < 1:
        raise ValueError(f"The num_sampled_frames must be greater than 1.")
    if sample_stride is not None and sample_stride < 1:
        raise ValueError(f"The sample_stride must be greater than 1.")
    if sample_stride is not None and sample_method not in ["random", "stride"]:
        raise ValueError(f"The sample_method must be random or stride when sample_stride is specified.")
    with video_reader(video_path, num_threads=2, **kwargs) as vr:
        if sample_method == "mid":
            sampled_frame_idx_list = [len(vr) // 2]
        elif sample_method == "uniform":
            sampled_frame_idx_list = np.linspace(0, len(vr), num_sampled_frames, endpoint=False, dtype=int)
        elif sample_method == "random":
            clip_length = min(len(vr), (num_sampled_frames - 1) * sample_stride + 1)
            start_idx = random.randint(0, len(vr) - clip_length)
            sampled_frame_idx_list = np.linspace(start_idx, start_idx + clip_length - 1, num_sampled_frames, dtype=int)
        elif sample_method == "stride":
            sampled_frame_idx_list = np.arange(0, len(vr), sample_stride)
        elif sample_method == "first":
            sampled_frame_idx_list = [0]
        elif sample_method == "last":
            sampled_frame_idx_list = [len(vr) - 1]
        elif sample_method == "keyframe":
            sampled_frame_idx_list, final_frame_index = get_keyframe_index(video_path)
        elif sample_method == "keyframe+first":  # keyframe + the first second
            sampled_frame_idx_list, final_frame_index = get_keyframe_index(video_path)
            if len(sampled_frame_idx_list) == 1 or sampled_frame_idx_list[1] > 1 * vr.get_avg_fps():
                if int(1 * vr.get_avg_fps()) > len(vr):
                    raise ValueError(f"The duration of {video_path} is less than 1s.")
                sampled_frame_idx_list.insert(1, int(1 * vr.get_avg_fps()))
        elif sample_method == "keyframe+last":  # keyframe + the last frame
            sampled_frame_idx_list, final_frame_index = get_keyframe_index(video_path)
            if sampled_frame_idx_list[-1] != (len(vr) - 1):
                sampled_frame_idx_list.append(len(vr) - 1)
        else:
            raise ValueError(f"The sample_method must be within {ALL_FRAME_SAMPLE_METHODS}.")
        if "keyframe" in sample_method:
            if final_frame_index != len(vr):
                raise ValueError(f"The keyframe index list is not accurate. Please check the video {video_path}.")
        sampled_frame_list = vr.get_batch(sampled_frame_idx_list).asnumpy()
        sampled_frame_list = [Image.fromarray(frame) for frame in sampled_frame_list]

        return list(sampled_frame_idx_list), sampled_frame_list
