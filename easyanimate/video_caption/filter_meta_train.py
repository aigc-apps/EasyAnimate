import argparse
import os

import pandas as pd
from natsort import natsorted

from utils.logger import logger
from utils.filter import filter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--caption_metadata_path", type=str, default=None, help="The path to the video quality metadata (csv/jsonl)."
    )
    parser.add_argument(
        "--video_path_column",
        type=str,
        default="video_path",
        help="The column contains the video path (an absolute path or a relative path w.r.t the video_folder).",
    )
    parser.add_argument("--video_folder", type=str, default="", help="The video folder.")
    parser.add_argument(
        "--basic_metadata_path", type=str, default=None, help="The path to the basic metadata (csv/jsonl)."
    )
    parser.add_argument("--min_resolution", type=float, default=720*1280, help="The resolution threshold.")
    parser.add_argument("--min_duration", type=float, default=-1, help="The minimum duration.")
    parser.add_argument("--max_duration", type=float, default=-1, help="The maximum duration.")
    parser.add_argument(
        "--asethetic_score_metadata_path", type=str, default=None, help="The path to the video quality metadata (csv/jsonl)."
    )
    parser.add_argument("--min_asethetic_score", type=float, default=4.0, help="The asethetic score threshold.")
    parser.add_argument(
        "--asethetic_score_siglip_metadata_path", type=str, default=None, help="The path to the video quality (SigLIP) metadata (csv/jsonl)."
    )
    parser.add_argument("--min_asethetic_score_siglip", type=float, default=4.0, help="The asethetic score (SigLIP) threshold.")
    parser.add_argument(
        "--text_score_metadata_path", type=str, default=None, help="The path to the video text score metadata (csv/jsonl)."
    )
    parser.add_argument("--min_text_score", type=float, default=0.02, help="The text threshold.")
    parser.add_argument(
        "--motion_score_metadata_path", type=str, default=None, help="The path to the video motion score metadata (csv/jsonl)."
    )
    parser.add_argument("--min_motion_score", type=float, default=2, help="The motion threshold.")
    parser.add_argument(
        "--videoclipxl_score_metadata_path", type=str, default=None, help="The path to the video-caption VideoCLIPXL score metadata (csv/jsonl)."
    )
    parser.add_argument("--min_videoclipxl_score", type=float, default=0.20, help="The VideoCLIPXL score threshold.")
    parser.add_argument("--saved_path", type=str, required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    raw_caption_df = pd.read_json(args.caption_metadata_path, lines=True)
    video_path_list = raw_caption_df[args.video_path_column].to_list()
    filtered_video_path_list = filter(
        video_path_list,
        basic_metadata_path=args.basic_metadata_path,
        min_resolution=args.min_resolution,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        asethetic_score_metadata_path=args.asethetic_score_metadata_path,
        min_asethetic_score=args.min_asethetic_score,
        asethetic_score_siglip_metadata_path=args.asethetic_score_siglip_metadata_path,
        min_asethetic_score_siglip=args.min_asethetic_score_siglip,
        text_score_metadata_path=args.text_score_metadata_path,
        min_text_score=args.min_text_score,
        motion_score_metadata_path=args.motion_score_metadata_path,
        min_motion_score=args.min_motion_score,
        videoclipxl_score_metadata_path=args.videoclipxl_score_metadata_path,
        min_videoclipxl_score=args.min_videoclipxl_score,
        video_path_column=args.video_path_column
    )
    filtered_video_path_list = natsorted(filtered_video_path_list)
    filtered_caption_df = raw_caption_df[raw_caption_df[args.video_path_column].isin(filtered_video_path_list)]
    train_df = filtered_caption_df.rename(columns={"video_path": "file_path", "caption": "text"})
    train_df["file_path"] = train_df["file_path"].map(lambda x: os.path.join(args.video_folder, x))
    train_df["type"] = "video"
    train_df.to_json(args.saved_path, orient="records", force_ascii=False, indent=2)
    logger.info(f"The final train file with {len(train_df)} videos are saved to {args.saved_path}.")


if __name__ == "__main__":
    main()