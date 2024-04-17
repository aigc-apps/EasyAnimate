from pathlib import Path

import pandas as pd
from func_timeout import FunctionTimedOut, func_timeout
from torch.utils.data import DataLoader, Dataset

from utils.logger import logger
from utils.video_utils import get_video_path_list, extract_frames

ALL_VIDEO_EXT = set(["mp4", "webm", "mkv", "avi", "flv", "mov"])
VIDEO_READER_TIMEOUT = 10


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) != 0:
        return {k: [item[k] for item in batch] for k in batch[0].keys()}
    return {}


class VideoDataset(Dataset):
    def __init__(
        self,
        video_path_list=None,
        video_folder=None,
        video_metadata_path=None,
        video_path_column=None,
        sample_method="mid",
        num_sampled_frames=1,
        num_sample_stride=None,
    ):
        self.video_path_column = video_path_column
        self.video_folder = video_folder
        self.sample_method = sample_method
        self.num_sampled_frames = num_sampled_frames
        self.num_sample_stride = num_sample_stride

        if video_path_list is not None:
            self.video_path_list = video_path_list
            self.metadata_df = pd.DataFrame({video_path_column: self.video_path_list})
        else:
            self.video_path_list = get_video_path_list(
                video_folder=video_folder,
                video_metadata_path=video_metadata_path,
                video_path_column=video_path_column
            )

    def __getitem__(self, index):
        # video_path = os.path.join(self.video_folder, str(self.video_path_list[index]))
        video_path = self.video_path_list[index]
        try:
            sample_args = (video_path, self.sample_method, self.num_sampled_frames, self.num_sample_stride)
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
            "video_path": Path(video_path).name,
            "sampled_frame_idx": sampled_frame_idx_list,
            "sampled_frame": sampled_frame_list,
        }

        return item

    def __len__(self):
        return len(self.video_path_list)


if __name__ == "__main__":
    video_folder = "your_video_folder"
    video_dataset = VideoDataset(video_folder=video_folder)

    video_dataloader = DataLoader(
        video_dataset, batch_size=16, num_workers=16, collate_fn=collate_fn
    )
    for idx, batch in enumerate(video_dataloader):
        if len(batch) != 0:
            print(batch["video_path"], batch["sampled_frame_idx"], len(batch["video_path"]))