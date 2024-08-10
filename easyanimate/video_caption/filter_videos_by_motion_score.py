import ast
import argparse
import gc
import os
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from natsort import natsorted
from tqdm import tqdm

from utils.logger import logger
from utils.video_utils import get_video_path_list

def parse_args():
    parser = argparse.ArgumentParser(description="Filter the motion score of the videos.")
    parser.add_argument(
        "--motion_score_metadata_path", type=str, default=None, help="The path to the video dataset metadata (csv/jsonl)."
    )
    parser.add_argument("--low_motion_score_threshold", type=float, default=3.0, help="The low motion score threshold.")
    parser.add_argument("--high_motion_score_threshold", type=float, default=8.0, help="The high motion score threshold.")
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not (args.saved_path.endswith(".csv") or args.saved_path.endswith(".jsonl")):
        raise ValueError("The saved_path must end with .csv or .jsonl.")
    
    if args.motion_score_metadata_path is not None:
        if args.motion_score_metadata_path.endswith(".csv"):
            motion_score_df = pd.read_csv(args.motion_score_metadata_path)
        elif args.motion_score_metadata_path.endswith(".jsonl"):
            motion_score_df = pd.read_json(args.motion_score_metadata_path, lines=True)

        filtered_motion_score_df = motion_score_df[motion_score_df["motion_score"] > args.low_motion_score_threshold]
        filtered_motion_score_df = filtered_motion_score_df[motion_score_df["motion_score"] < args.high_motion_score_threshold]

    if args.saved_path.endswith(".csv"):
        header = False if os.path.exists(args.saved_path) else True
        filtered_motion_score_df.to_csv(args.saved_path, header=header, index=False, mode="a")
    elif args.saved_path.endswith(".jsonl"):
        filtered_motion_score_df.to_json(args.saved_path, orient="records", lines=True, mode="a")
    logger.info(f"Save result to {args.saved_path}.")


if __name__ == "__main__":
    main()