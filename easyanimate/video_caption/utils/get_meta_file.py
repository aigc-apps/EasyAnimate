import argparse
import os

import pandas as pd
from natsort import natsorted

from .logger import logger

ALL_VIDEO_EXT = set(["mp4", "webm", "mkv", "avi", "flv", "mov", "rmvb"])
ALL_IMGAE_EXT = set(["png", "webp", "jpg", "jpeg", "bmp", "gif"])


def get_relative_file_paths(directory, recursive=False, ext_set=None):
    """Get the relative paths of subfiles (recursively) in the directory that match the extension set.
    """
    if not recursive:
        for entry in os.scandir(directory):
            if entry.is_file():
                file_name = entry.name
                if ext_set is not None:
                    ext = os.path.splitext(file_name)[1][1:].lower()
                    if ext in ext_set:
                        yield file_name
                else:
                    yield file_name
    else:
        for root, _, files in os.walk(directory):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                if ext_set is not None:
                    ext = os.path.splitext(file)[1][1:].lower()
                    if ext in ext_set:
                        yield relative_path
                else:
                    yield relative_path


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
    if args.image_folder is None and not os.path.exists(args.video_folder):
        raise ValueError(f"The video_folder {args.video_folder} does not exist.")
    if args.video_folder is None and not os.path.exists(args.image_folder):
        raise ValueError(f"The image_folder {args.image_folder} does not exist.")

    # Use the path name instead of the file name as video_path/image_path (unique ID).
    if args.video_folder is not None:
        video_path_list = list(get_relative_file_paths(args.video_folder, recursive=args.recursive, ext_set=ALL_VIDEO_EXT))
        video_path_list = natsorted(video_path_list)
        meta_file_df = pd.DataFrame({args.video_path_column: video_path_list})
    
    if args.image_folder is not None:
        image_path_list = list(get_relative_file_paths(args.image_folder, recursive=args.recursive, ext_set=ALL_IMGAE_EXT))
        image_path_list = natsorted(image_path_list)
        meta_file_df = pd.DataFrame({args.image_path_column: image_path_list})

    logger.info(f"{len(meta_file_df)} files in total. Save the result to {args.saved_path}.")
    meta_file_df.to_json(args.saved_path, orient="records", lines=True)


if __name__ == "__main__":
    main()