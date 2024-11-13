import os
from pathlib import Path
from typing import Optional

from func_timeout import FunctionTimedOut, func_timeout
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .logger import logger
from .video_utils import extract_frames


ALL_VIDEO_EXT = set([".mp4", ".webm", ".mkv", ".avi", ".flv", ".mov", ".ts"])
VIDEO_READER_TIMEOUT = 300


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) != 0:
        return {k: [item[k] for item in batch] for k in batch[0].keys()}
    return {}


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_inputs: dict[str, list[str]],
        video_folder: Optional[str] = None,
        video_path_column: str = "video_path",
        text_column: Optional[str] = None,
        sample_method: str = "mid",
        num_sampled_frames: int = 1,
        sample_stride: Optional[int] = None
    ):
        length = len(dataset_inputs[list(dataset_inputs.keys())[0]])
        if not all(len(v) == length for v in dataset_inputs.values()):
            raise ValueError("All values in the dataset_inputs must have the same length.")
        
        self.video_path_column = video_path_column
        self.video_folder = video_folder
        self.video_path_list = dataset_inputs[video_path_column]
        if self.video_folder is not None:
            self.video_path_list = [os.path.join(self.video_folder, video_path) for video_path in self.video_path_list]
        self.text_column = text_column
        self.text_list = dataset_inputs[self.text_column] if self.text_column is not None else None

        self.sample_method = sample_method
        self.num_sampled_frames = num_sampled_frames
        self.sample_stride = sample_stride

    def __getitem__(self, index):
        video_path = self.video_path_list[index]
        if self.sample_method == "image":
            try:
                sampled_frame_idx_list = None
                with open(video_path, "rb") as f:
                    sampled_frame_list = [Image.open(f).convert("RGB")]
            except Exception as e:
                logger.warning(f"Failed to extract frames from video {video_path}. Error is {e}.")
                return None
        else:
            # It is a trick to deal with decord hanging when reading some abnormal videos.
            try:
                sample_args = (video_path, self.sample_method, self.num_sampled_frames, self.sample_stride)
                sampled_frame_idx_list, sampled_frame_list = func_timeout(
                    VIDEO_READER_TIMEOUT, extract_frames, args=sample_args
                )
            except FunctionTimedOut:
                logger.warning(f"Read {video_path} timeout.")
                return None
            except Exception as e:
                logger.warning(f"Failed to extract frames from video {video_path}. Error is {e}.")
                return None
            
        item = {
            "path": video_path,
            "sampled_frame_idx": sampled_frame_idx_list,
            "sampled_frame": sampled_frame_list,
        }
        if self.text_list is not None:
            item["text"] = self.text_list[index]

        return item

    def __len__(self):
        return len(self.video_path_list)


if __name__ == "__main__":
    video_folder = Path("your_video_folder")
    video_path_list = []
    for ext in ALL_VIDEO_EXT:
        video_path_list += [str(file.relative_to(video_folder)) for file in video_folder.glob(f"*.{ext}")]

    video_dataset = VideoDataset(dataset_inputs={"video_path": video_path_list})
    video_dataloader = DataLoader(
        video_dataset, batch_size=16, num_workers=16, collate_fn=collate_fn
    )
    for idx, batch in enumerate(video_dataloader):
        if len(batch) != 0:
            print(batch["video_path"], batch["sampled_frame_idx"], len(batch["video_path"]))