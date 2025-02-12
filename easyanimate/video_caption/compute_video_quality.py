import argparse
import os

import pandas as pd
from accelerate import PartialState
from accelerate.utils import gather_object
from natsort import index_natsorted
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.image_evaluator as image_evaluator
import utils.video_evaluator as video_evaluator
from utils.filter import filter
from utils.logger import logger
from utils.video_dataset import VideoDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Compute scores of uniform sampled frames from videos.")
    parser.add_argument(
        "--video_metadata_path", type=str, default=None, help="The path to the video dataset metadata (csv/jsonl)."
    )
    parser.add_argument(
        "--video_path_column",
        type=str,
        default="video_path",
        help="The column contains the video path (an absolute path or a relative path w.r.t the video_folder).",
    )
    parser.add_argument("--video_folder", type=str, default="", help="The video folder.")
    parser.add_argument("--caption_column", type=str, default=None, help="The column contains the caption.")
    parser.add_argument(
        "--frame_sample_method",
        type=str,
        choices=["mid", "uniform", "image"],
        default="uniform",
    )
    parser.add_argument("--num_sampled_frames", type=int, default=8, help="The number of sampled frames.")
    parser.add_argument("--metrics", nargs="+", type=str, required=True, help="The evaluation metric(s) for generated images.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for the video dataset.")
    parser.add_argument("--num_workers", type=int, default=1, help="The number of workers for the video dataset.")
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--saved_freq", type=int, default=1, help="The frequency to save the output results.")

    parser.add_argument("--basic_metadata_path", type=str, default=None, help="The path to the basic metadata (csv/jsonl).")
    parser.add_argument("--min_resolution", type=float, default=0, help="The resolution threshold.")
    parser.add_argument("--min_duration", type=float, default=-1, help="The minimum duration.")
    parser.add_argument("--max_duration", type=float, default=-1, help="The maximum duration.")
    parser.add_argument(
        "--text_score_metadata_path", type=str, default=None, help="The path to the video text score metadata (csv/jsonl)."
    )
    parser.add_argument("--min_text_score", type=float, default=0.02, help="The text threshold.")
    parser.add_argument(
        "--motion_score_metadata_path", type=str, default=None, help="The path to the video motion score metadata (csv/jsonl)."
    )
    parser.add_argument("--min_motion_score", type=float, default=2, help="The minimum motion threshold.")
    parser.add_argument("--max_motion_score", type=float, default=999999, help="The maximum motion threshold.")
    parser.add_argument(
        "--semantic_consistency_score_metadata_path",
        nargs="+",
        type=str,
        default=None,
        help="The path to the semantic consistency metadata (csv/jsonl)."
    )
    parser.add_argument(
        "--min_semantic_consistency_score", type=float, default=0.80, help="The semantic consistency score threshold."
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    if args.video_metadata_path.endswith(".csv"):
        video_metadata_df = pd.read_csv(args.video_metadata_path)
    elif args.video_metadata_path.endswith(".jsonl"):
        video_metadata_df = pd.read_json(args.video_metadata_path, lines=True)
    else:
        raise ValueError("The video_metadata_path must end with .csv or .jsonl.")

    if not (args.saved_path.endswith(".csv") or args.saved_path.endswith(".jsonl")):
        raise ValueError("The saved_path must end with .csv or .jsonl.")
    
    if os.path.exists(args.saved_path):
        if args.saved_path.endswith(".csv"):
            saved_metadata_df = pd.read_csv(args.saved_path)
        elif args.saved_path.endswith(".jsonl"):
            saved_metadata_df = pd.read_json(args.saved_path, lines=True)

        # Filter out the unprocessed video-caption pairs by setting the indicator=True.
        merged_df = video_metadata_df.merge(saved_metadata_df, on=args.video_path_column, how="outer", indicator=True)
        video_metadata_df = merged_df[merged_df["_merge"] == "left_only"]
        # Sorting to guarantee the same result for each process.
        video_metadata_df = video_metadata_df.iloc[index_natsorted(video_metadata_df[args.video_path_column])].reset_index(drop=True)
        if args.caption_column is None:
            video_metadata_df = video_metadata_df[[args.video_path_column]]
        else:
            video_metadata_df = video_metadata_df[[args.video_path_column, args.caption_column + "_x"]]
            video_metadata_df.rename(columns={args.caption_column + "_x": args.caption_column}, inplace=True)
        logger.info(f"Resume from {args.saved_path}: {len(saved_metadata_df)} processed and {len(video_metadata_df)} to be processed.")
    
    video_path_list = video_metadata_df[args.video_path_column].tolist()
    video_path_list = filter(
        video_path_list,
        basic_metadata_path=args.basic_metadata_path,
        min_resolution=args.min_resolution,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        text_score_metadata_path=args.text_score_metadata_path,
        min_text_score=args.min_text_score,
        motion_score_metadata_path=args.motion_score_metadata_path,
        min_motion_score=args.min_motion_score,
        max_motion_score=args.max_motion_score,
        semantic_consistency_score_metadata_path=args.semantic_consistency_score_metadata_path,
        min_semantic_consistency_score=args.min_semantic_consistency_score,
        video_path_column=args.video_path_column
    )
    video_metadata_df = video_metadata_df[video_metadata_df[args.video_path_column].isin(video_path_list)]

    state = PartialState()
    metric_fns = []
    for metric in args.metrics:
        if hasattr(image_evaluator, metric):  # frame-wise
            if state.is_main_process:
                logger.info("Initializing frame-wise evaluator metrics...")
                # Check if the model is downloaded in the main process.
                getattr(image_evaluator, metric)(device="cpu")
            state.wait_for_everyone()
            metric_fns.append(getattr(image_evaluator, metric)(device=state.device))
        else:  # video-wise
            if state.is_main_process:
                logger.info("Initializing video-wise evaluator metrics...")
                # Check if the model is downloaded in the main process.
                getattr(video_evaluator, metric)(device="cpu")
            state.wait_for_everyone()
            metric_fns.append(getattr(video_evaluator, metric)(device=state.device))

    result_dict = {args.video_path_column: [], "sample_frame_idx": []}
    for metric in metric_fns:
        result_dict[str(metric)] = []
    if args.caption_column is not None:
        result_dict[args.caption_column] = []

    if args.frame_sample_method == "image":
        logger.warning("Set args.num_sampled_frames to 1 since args.frame_sample_method is image.")
        args.num_sampled_frames = 1
    
    index = len(video_metadata_df) - len(video_metadata_df) % state.num_processes
    # Avoid the NCCL timeout in the final gather operation.
    logger.info(
        f"Drop the last {len(video_metadata_df) % state.num_processes} videos "
        "to ensure each process handles the same number of videos."
    )
    video_metadata_df = video_metadata_df.iloc[:index]
    logger.info(f"{len(video_metadata_df)} videos are to be processed.")

    video_metadata_list = video_metadata_df.to_dict(orient='list')
    with state.split_between_processes(video_metadata_list) as splitted_video_metadata:
        video_dataset = VideoDataset(
            dataset_inputs=splitted_video_metadata,
            video_folder=args.video_folder,
            video_path_column=args.video_path_column,
            text_column=args.caption_column,
            sample_method=args.frame_sample_method,
            num_sampled_frames=args.num_sampled_frames
        )
        video_loader = DataLoader(video_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

        for idx, batch in enumerate(tqdm(video_loader)):
            if len(batch) > 0:
                batch_video_path = batch["path"]
                result_dict["sample_frame_idx"].extend(batch["sampled_frame_idx"])
                batch_frame = batch["sampled_frame"]  # [batch_size, num_sampled_frames, H, W, C]
                batch_caption = None
                if args.caption_column is not None:
                    batch_caption = batch["text"]
                    result_dict["caption"].extend(batch_caption)
                # Compute the quality.
                for i, metric in enumerate(args.metrics):
                    quality_scores = metric_fns[i](batch_frame, batch_caption)
                    if isinstance(quality_scores[0], list):  # frame-wise
                        quality_scores = [
                            [round(score, 5) for score in inner_list]
                            for inner_list in quality_scores
                        ]
                    else:  # video-wise
                        quality_scores = [round(score, 5) for score in quality_scores]
                    result_dict[str(metric_fns[i])].extend(quality_scores)
        
                if args.video_folder == "":
                    saved_video_path_list = batch_video_path
                else:
                    saved_video_path_list = [os.path.relpath(video_path, args.video_folder) for video_path in batch_video_path]
                result_dict[args.video_path_column].extend(saved_video_path_list)

            # Save the metadata in the main process every saved_freq.
            if (idx % args.saved_freq) == 0 or idx == len(video_loader) - 1:
                state.wait_for_everyone()
                gathered_result_dict = {k: gather_object(v) for k, v in result_dict.items()}
                if state.is_main_process and len(gathered_result_dict[args.video_path_column]) != 0:
                    result_df = pd.DataFrame(gathered_result_dict)
                    # Append is not supported (oss).
                    if args.saved_path.endswith(".csv"):
                        if os.path.exists(args.saved_path):
                            saved_df = pd.read_csv(args.saved_path)
                            result_df = pd.concat([saved_df, result_df], ignore_index=True)
                        result_df.to_csv(args.saved_path, index=False)
                    elif args.saved_path.endswith(".jsonl"):
                        if os.path.exists(args.saved_path):
                            saved_df = pd.read_json(args.saved_path, orient="records", lines=True)
                            result_df = pd.concat([saved_df, result_df], ignore_index=True)
                        result_df.to_json(args.saved_path, orient="records", lines=True, force_ascii=False)
                    logger.info(f"Save result to {args.saved_path}.")
                for k in result_dict.keys():
                    result_dict[k] = []

if __name__ == "__main__":
    main()
