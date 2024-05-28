import argparse
import re
import os

import pandas as pd
from accelerate import PartialState
from accelerate.utils import gather_object
from natsort import natsorted
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.image_evaluator as image_evaluator
from utils.logger import logger
from utils.video_dataset import VideoDataset, collate_fn
from utils.video_utils import get_video_path_list


def camel2snake(s: str) -> str:
    """Convert camel case to snake case."""
    if not re.match("^[A-Z]+$", s):
        pattern = re.compile(r"(?<!^)(?=[A-Z])")
        return pattern.sub("_", s).lower()
    return s


def parse_args():
    parser = argparse.ArgumentParser(description="Compute scores of uniform sampled frames from videos.")
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
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column contains the caption.",
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=4,
        help="num_sampled_frames",
    )
    parser.add_argument("--metrics", nargs="+", type=str, required=True, help="The evaluation metric(s) for generated images.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        required=False,
        help="The batch size for the video dataset.",
    )
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--saved_freq", type=int, default=1000, help="The frequency to save the output results.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.batch_size > 1

    video_path_list = get_video_path_list(
        video_folder=args.video_folder,
        video_metadata_path=args.video_metadata_path,
        video_path_column=args.video_path_column
    )

    if not (args.saved_path.endswith(".csv") or args.saved_path.endswith(".jsonl")):
        raise ValueError("The saved_path must end with .csv or .jsonl.")

    caption_list = None
    if args.video_metadata_path is not None and args.caption_column is not None:
        if args.video_metadata_path.endswith(".csv"):
            video_metadata_df = pd.read_csv(args.video_metadata_path)
        elif args.video_metadata_path.endswith(".jsonl"):
            video_metadata_df = pd.read_json(args.video_metadata_path, lines=True)
        else:
            raise ValueError("The video_metadata_path must end with .csv or .jsonl.")
        caption_list = video_metadata_df[args.caption_column].tolist()

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

    logger.info("Initializing evaluator metrics...")
    state = PartialState()
    metric_fns = [getattr(image_evaluator, metric)(device=state.device) for metric in args.metrics]

    # The workaround can be removed after https://github.com/huggingface/accelerate/pull/2781 is released.
    index = len(video_path_list) - len(video_path_list) % state.num_processes
    logger.info(f"Drop {len(video_path_list) % state.num_processes} videos to avoid duplicates in state.split_between_processes.")
    video_path_list = video_path_list[:index]

    result_dict = {args.video_path_column: [], "sample_frame_idx": []}
    for metric in args.metrics:
        result_dict[camel2snake(metric)] = []
    
    with state.split_between_processes(video_path_list) as splitted_video_path_list:
        video_dataset = VideoDataset(
            video_path_list=splitted_video_path_list,
            sample_method="uniform",
            num_sampled_frames=args.num_sampled_frames
        )
        video_loader = DataLoader(video_dataset, batch_size=args.batch_size, num_workers=4, collate_fn=collate_fn)
        for idx, batch in enumerate(tqdm(video_loader)):
            if len(batch) == 0:
                continue
            batch_video_path = batch[args.video_path_column]
            result_dict["sample_frame_idx"].extend(batch["sampled_frame_idx"])
            # [batch_size, num_sampled_frames, H, W, C] => [batch_size * num_sampled_frames, H, W, C].
            batch_frame = []
            for item_sampled_frame in batch["sampled_frame"]:
                batch_frame.extend([frame for frame in item_sampled_frame])
            batch_caption = None
            if caption_list is not None:
                batch_caption = caption_list[i : i + args.batch_size]
            # Compute the frame quality.
            for i, metric in enumerate(args.metrics):
                # [batch_size * num_sampled_frames] => [batch_size, num_sampled_frames]
                quality_scores = metric_fns[i](batch_frame, batch_caption)
                quality_scores = [round(score, 5) for score in quality_scores]
                quality_scores = [quality_scores[j:j + args.num_sampled_frames] for j in range(0, len(quality_scores), args.num_sampled_frames)]
                result_dict[camel2snake(metric)].extend(quality_scores)
            
            saved_video_path_list = [os.path.basename(video_path) for video_path in batch_video_path]
            result_dict[args.video_path_column].extend(saved_video_path_list)

            # Save the metadata in the main process every saved_freq.
            if (idx != 0) and (idx % args.saved_freq == 0):
                state.wait_for_everyone()
                gathered_result_dict = {k: gather_object(v) for k, v in result_dict.items()}
                if state.is_main_process:
                    result_df = pd.DataFrame(gathered_result_dict)
                    if args.saved_path.endswith(".csv"):
                        header = False if os.path.exists(args.saved_path) else True
                        result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
                    elif args.saved_path.endswith(".jsonl"):
                        result_df.to_json(args.saved_path, orient="records", lines=True, mode="a")
                    logger.info(f"Save result to {args.saved_path}.")
                for k in result_dict.keys():
                    result_dict[k] = []

    # Wait for all processes to finish and gather the final result.
    state.wait_for_everyone()
    gathered_result_dict = {k: gather_object(v) for k, v in result_dict.items()}
    # Save the metadata in the main process.
    if state.is_main_process:
        result_df = pd.DataFrame(gathered_result_dict)
        if len(gathered_result_dict[args.video_path_column]) != 0:
            result_df = pd.DataFrame(gathered_result_dict)
            if args.saved_path.endswith(".csv"):
                header = False if os.path.exists(args.saved_path) else True
                result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
            elif args.saved_path.endswith(".jsonl"):
                result_df.to_json(args.saved_path, orient="records", lines=True, mode="a")
            logger.info(f"Save the final result to {args.saved_path}.")


if __name__ == "__main__":
    main()
