import argparse
from pathlib import Path

import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from .logger import logger


ALL_VIDEO_EXT = set(["mp4", "webm", "mkv", "avi", "flv", "mov"])
ALL_IMGAE_EXT = set(["png", "webp", "jpg", "jpeg", "bmp", "gif"])


def parse_args():
    parser = argparse.ArgumentParser(description="Compute scores of uniform sampled frames from videos.")
    parser.add_argument(
        "--image_path_column",
        type=str,
        default="image_path",
        help="The column contains the image path (an absolute path or a relative path w.r.t the image_folder).",
    )
    parser.add_argument("--image_folder", type=str, default=None, help="The video folder.")
    parser.add_argument(
        "--video_path_column",
        type=str,
        default="video_path",
        help="The column contains the video path (an absolute path or a relative path w.r.t the video_folder).",
    )
    parser.add_argument("--video_folder", type=str, default=None, help="The video folder.")
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--recursive", action="store_true", help="Whether to search sub-folders recursively.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.video_folder is None and args.image_folder is None:
        raise ValueError("Either video_folder or image_folder should be specified in the arguments.")
    if args.video_folder is not None and args.image_folder is not None:
        raise ValueError("Both video_folder and image_folder can not be specified in the arguments at the same time.")

    # Use the path name instead of the file name as video_path/image_path (unique ID).
    if args.video_folder is not None:
        video_path_list = []
        video_folder = Path(args.video_folder)
        for ext in tqdm(list(ALL_VIDEO_EXT)):
            if args.recursive:
                video_path_list += [str(file.relative_to(video_folder)) for file in video_folder.rglob(f"*.{ext}")]
            else:
                video_path_list += [str(file.relative_to(video_folder)) for file in video_folder.glob(f"*.{ext}")]
        video_path_list = natsorted(video_path_list)
        meta_file_df = pd.DataFrame({args.video_path_column: video_path_list})
    
    if args.image_folder is not None:
        image_path_list = []
        image_folder = Path(args.image_folder)
        for ext in tqdm(list(ALL_IMGAE_EXT)):
            if args.recursive:
                image_path_list += [str(file.relative_to(image_folder)) for file in image_folder.rglob(f"*.{ext}")]
            else:
                image_path_list += [str(file.relative_to(image_folder)) for file in image_folder.glob(f"*.{ext}")]
        image_path_list = natsorted(image_path_list)
        meta_file_df = pd.DataFrame({args.image_path_column: image_path_list})

    logger.info(f"{len(meta_file_df)} files in total. Save the result to {args.saved_path}.")
    meta_file_df.to_json(args.saved_path, orient="records", lines=True)


if __name__ == "__main__":
    main()