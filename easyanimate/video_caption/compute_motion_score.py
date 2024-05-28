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


@contextmanager
def VideoCapture(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        yield cap
    finally:
        cap.release()
        del cap
        gc.collect()


def compute_motion_score(video_path):
    video_motion_scores = []
    sampling_fps = 2

    try:
        with VideoCapture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            valid_fps = min(max(sampling_fps, 1), fps)
            frame_interval = int(fps / valid_fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # if cannot get the second frame, use the last one
            frame_interval = min(frame_interval, total_frames - 1)

            prev_frame = None
            frame_count = -1
            while cap.isOpened():
                ret, frame = cap.read()
                frame_count += 1

                if not ret:
                    break

                # skip middle frames
                if frame_count % frame_interval != 0:
                    continue

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_frame is None:
                    prev_frame = gray_frame
                    continue

                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame,
                    gray_frame,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                frame_motion_score = np.mean(mag)
                video_motion_scores.append(frame_motion_score)
                prev_frame = gray_frame

            video_meta_info = {
                "video_path": Path(video_path).name,
                "motion_score": round(float(np.mean(video_motion_scores)), 5),
            }
            return video_meta_info

    except Exception as e:
        print(f"Compute motion score for video {video_path} with error: {e}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the motion score of the videos.")
    parser.add_argument("--video_folder", type=str, default="", help="The video folder.")
    parser.add_argument(
        "--video_metadata_path", type=str, default=None, help="The path to the video dataset metadata (csv/jsonl)."
    )
    parser.add_argument(
        "--video_path_column",
        type=str,
        default="video_path",
        help="The column contains the video path (an absolute path or a relative path w.r.t the video_folder).",
    )
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--saved_freq", type=int, default=100, help="The frequency to save the output results.")
    parser.add_argument("--n_jobs", type=int, default=1, help="The number of concurrent processes.")

    parser.add_argument(
        "--asethetic_score_metadata_path", type=str, default=None, help="The path to the video quality metadata (csv/jsonl)."
    )
    parser.add_argument("--asethetic_score_threshold", type=float, default=4.0, help="The asethetic score threshold.")
    parser.add_argument(
        "--text_score_metadata_path", type=str, default=None, help="The path to the video text score metadata (csv/jsonl)."
    )
    parser.add_argument("--text_score_threshold", type=float, default=0.02, help="The text threshold.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    video_path_list = get_video_path_list(
        video_folder=args.video_folder,
        video_metadata_path=args.video_metadata_path,
        video_path_column=args.video_path_column
    )

    if not (args.saved_path.endswith(".csv") or args.saved_path.endswith(".jsonl")):
        raise ValueError("The saved_path must end with .csv or .jsonl.")
    
    if os.path.exists(args.saved_path):
        if args.saved_path.endswith(".csv"):
            saved_metadata_df = pd.read_csv(args.saved_path)
        elif args.saved_path.endswith(".jsonl"):
            saved_metadata_df = pd.read_json(args.saved_path, lines=True)
        saved_video_path_list = saved_metadata_df[args.video_path_column].tolist()
        saved_video_path_list = [os.path.join(args.video_folder, video_path) for video_path in saved_video_path_list]
        
        video_path_list = list(set(video_path_list).difference(set(saved_video_path_list)))
        # Sorting to guarantee the same result for each process.
        video_path_list = natsorted(video_path_list)
        logger.info(f"Resume from {args.saved_path}: {len(saved_video_path_list)} processed and {len(video_path_list)} to be processed.")
    
    if args.asethetic_score_metadata_path is not None:
        if args.asethetic_score_metadata_path.endswith(".csv"):
            asethetic_score_df = pd.read_csv(args.asethetic_score_metadata_path)
        elif args.asethetic_score_metadata_path.endswith(".jsonl"):
            asethetic_score_df = pd.read_json(args.asethetic_score_metadata_path, lines=True)

        # In pandas, csv will save lists as strings, whereas jsonl will not.
        asethetic_score_df["aesthetic_score"] = asethetic_score_df["aesthetic_score"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        asethetic_score_df["aesthetic_score_mean"] = asethetic_score_df["aesthetic_score"].apply(lambda x: sum(x) / len(x))
        filtered_asethetic_score_df = asethetic_score_df[asethetic_score_df["aesthetic_score_mean"] < args.asethetic_score_threshold]
        filtered_video_path_list = filtered_asethetic_score_df[args.video_path_column].tolist()
        filtered_video_path_list = [os.path.join(args.video_folder, video_path) for video_path in filtered_video_path_list]

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        # Sorting to guarantee the same result for each process.
        video_path_list = natsorted(video_path_list)
        logger.info(f"Load {args.asethetic_score_metadata_path} and filter {len(filtered_video_path_list)} videos.")
    
    if args.text_score_metadata_path is not None:
        if args.text_score_metadata_path.endswith(".csv"):
            text_score_df = pd.read_csv(args.text_score_metadata_path)
        elif args.text_score_metadata_path.endswith(".jsonl"):
            text_score_df = pd.read_json(args.text_score_metadata_path, lines=True)

        filtered_text_score_df = text_score_df[text_score_df["text_score"] > args.text_score_threshold]
        filtered_video_path_list = filtered_text_score_df[args.video_path_column].tolist()
        filtered_video_path_list = [os.path.join(args.video_folder, video_path) for video_path in filtered_video_path_list]

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        # Sorting to guarantee the same result for each process.
        video_path_list = natsorted(video_path_list)
        logger.info(f"Load {args.text_score_metadata_path} and filter {len(filtered_video_path_list)} videos.")

    for i in tqdm(range(0, len(video_path_list), args.saved_freq)):
        result_list = Parallel(n_jobs=args.n_jobs, backend="threading")(
            delayed(compute_motion_score)(video_path) for video_path in tqdm(video_path_list[i: i + args.saved_freq])
        )
        result_list = [result for result in result_list if result is not None]
        if len(result_list) == 0:
            continue

        result_df = pd.DataFrame(result_list)
        if args.saved_path.endswith(".csv"):
            header = False if os.path.exists(args.saved_path) else True
            result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
        elif args.saved_path.endswith(".jsonl"):
            result_df.to_json(args.saved_path, orient="records", lines=True, mode="a")
        logger.info(f"Save result to {args.saved_path}.")


if __name__ == "__main__":
    main()