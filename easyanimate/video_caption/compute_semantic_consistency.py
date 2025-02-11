import argparse
import os

import numpy as np
import pandas as pd
import torch
from accelerate import PartialState
from accelerate.utils import gather_object
from natsort import natsorted
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from utils.filter import filter
from utils.logger import logger
from utils.video_dataset import VideoDataset, collate_fn
from utils.video_utils import ALL_FRAME_SAMPLE_METHODS


ALL_MODEL_NAME = [
    "dinov2-small",
    "dinov2-base",
    "dinov2-large",
    "clip-vit-large-patch14",
    "clip-vit-base-patch32",
    "clip-vit-large-patch14-336",
]


def init_model(model_name, device):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return processor, model


def compute_adjacent_similarity(frame_features):
    frame_features /= frame_features.norm(dim=-1, keepdim=True)
    roll_frame_features = torch.roll(frame_features, shifts=-1, dims=0)
    similarity_matrix = frame_features.squeeze(dim=1).cpu().numpy() @ roll_frame_features.squeeze(dim=1).cpu().numpy().T

    return np.diag(similarity_matrix).tolist()[:-1]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the semantic consistency score across frames.")
    parser.add_argument(
        "--video_metadata_path", type=str, required=True, help="The path to the video dataset metadata (csv/jsonl)."
    )
    parser.add_argument(
        "--video_path_column",
        type=str,
        default="video_path",
        help="The column contains the video path (an absolute path or a relative path w.r.t the video_folder).",
    )
    parser.add_argument("--video_folder", type=str, default="", help="The video folder.")
    parser.add_argument(
        "--model_path", type=str, default="openai/clip-vit-large-patch14-336", help="The path to the DINO/CLIP model."
    )
    parser.add_argument("--frame_sample_method", type=str, choices=ALL_FRAME_SAMPLE_METHODS, default="keyframe+first")
    parser.add_argument("--num_sampled_frames", type=int, default=1, help="The number of sampled frames.")
    parser.add_argument("--sample_stride", type=int, default=None, help="The stride between two sampled frames.")
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for the video dataset.")
    parser.add_argument("--num_workers", type=int, default=1, help="The number of workers for the video dataset.")
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--saved_freq", type=int, default=1, help="The frequency to save the output results.")

    parser.add_argument("--basic_metadata_path", type=str, default=None, help="The path to the basic metadata (csv/jsonl).")
    parser.add_argument("--min_resolution", type=float, default=0, help="The resolution threshold.")
    parser.add_argument("--min_duration", type=float, default=-1, help="The minimum duration.")
    parser.add_argument("--max_duration", type=float, default=-1, help="The maximum duration.")
    parser.add_argument(
        "--aesthetic_score_metadata_path", type=str, default=None, help="The path to the video quality metadata (csv/jsonl)."
    )
    parser.add_argument("--min_aesthetic_score", type=float, default=4.0, help="The aesthetic score threshold.")
    parser.add_argument(
        "--aesthetic_score_siglip_metadata_path", type=str, default=None, help="The path to the video quality metadata (csv/jsonl)."
    )
    parser.add_argument("--min_aesthetic_score_siglip", type=float, default=4.0, help="The aesthetic score (SigLIP) threshold.")
    parser.add_argument(
        "--text_score_metadata_path", type=str, default=None, help="The path to the video text score metadata (csv/jsonl)."
    )
    parser.add_argument("--min_text_score", type=float, default=0.02, help="The text threshold.")
    parser.add_argument(
        "--motion_score_metadata_path", type=str, default=None, help="The path to the video motion score metadata (csv/jsonl)."
    )
    parser.add_argument("--min_motion_score", type=float, default=2, help="The motion threshold.")
    parser.add_argument("--max_motion_score", type=float, default=999999, help="The maximum motion threshold.")

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
    video_path_list = video_metadata_df[args.video_path_column].tolist()
    
    if not (args.saved_path.endswith(".csv") or args.saved_path.endswith(".jsonl")):
        raise ValueError("The saved_path must end with .csv or .jsonl.")
    
    if os.path.exists(args.saved_path):
        if args.saved_path.endswith(".csv"):
            saved_metadata_df = pd.read_csv(args.saved_path)
        elif args.saved_path.endswith(".jsonl"):
            saved_metadata_df = pd.read_json(args.saved_path, lines=True)
        saved_video_path_list = saved_metadata_df[args.video_path_column].tolist()
        video_path_list = list(set(video_path_list).difference(set(saved_video_path_list)))
        logger.info(f"Resume from {args.saved_path}: {len(saved_video_path_list)} processed and {len(video_path_list)} to be processed.")
    
    video_path_list = filter(
        video_path_list,
        basic_metadata_path=args.basic_metadata_path,
        min_resolution=args.min_resolution,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        aesthetic_score_metadata_path=args.aesthetic_score_metadata_path,
        min_aesthetic_score=args.min_aesthetic_score,
        aesthetic_score_siglip_metadata_path=args.aesthetic_score_siglip_metadata_path,
        min_aesthetic_score_siglip=args.min_aesthetic_score_siglip,
        text_score_metadata_path=args.text_score_metadata_path,
        min_text_score=args.min_text_score,
        motion_score_metadata_path=args.motion_score_metadata_path,
        min_motion_score=args.min_motion_score,
        max_motion_score=args.max_motion_score,
        video_path_column=args.video_path_column
    )
    # Sorting to guarantee the same result for each process.
    video_path_list = natsorted(video_path_list)

    if not any(name in args.model_path for name in ALL_MODEL_NAME):
        raise ValueError(f"The model_path should be among the following list: {ALL_MODEL_NAME}.")

    state = PartialState()
    if state.is_main_process:
        # Check if the model is downloaded in the main process.
        processor, model = init_model(args.model_path, "cpu")
    state.wait_for_everyone()
    processor, model = init_model(args.model_path, state.device)

    index = len(video_path_list) - len(video_path_list) % state.num_processes
    # Avoid the NCCL timeout in the final gather operation.
    logger.warning(
        f"Drop the last {len(video_path_list) % state.num_processes} videos "
        "to ensure each process handles the same number of videos."
    )
    video_path_list = video_path_list[:index]
    logger.info(f"{len(video_path_list)} videos are to be processed.")

    result_dict = {
        args.video_path_column: [],
        "similarity_cross_frame": [],
        "similarity_mean": [],
        "sample_frame_idx": [],
    }
    with state.split_between_processes(video_path_list) as splitted_video_path_list:
        video_dataset = VideoDataset(
            dataset_inputs={args.video_path_column: splitted_video_path_list},
            video_folder=args.video_folder,
            video_path_column=args.video_path_column,
            sample_method=args.frame_sample_method,
            num_sampled_frames=args.num_sampled_frames,
            sample_stride=args.sample_stride,
        )
        video_loader = DataLoader(video_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

        for idx, batch in enumerate(tqdm(video_loader)):
            if len(batch) > 0:
                batch_video_path = []
                batch_frame = []
                batch_sampled_frame_idx = []
                # At least two frames are required to calculate cross-frame semantic consistency.
                for path, frame, frame_idx in zip(batch["path"], batch["sampled_frame"], batch["sampled_frame_idx"]):
                    if len(frame) > 1:
                        batch_video_path.append(path)
                        batch_frame.append(frame)
                        batch_sampled_frame_idx.append(frame_idx)
                    else:
                        logger.warning(f"Skip {path} because it only has {len(frame)} frames.")

                frame_num_list = [len(video_frames) for video_frames in batch_frame]
                # [B, T, H, W, C] => [(B * T), H, W, C]
                reshaped_batch_frame = [frame for video_frames in batch_frame for frame in video_frames]
                with torch.no_grad():
                    inputs = processor(images=reshaped_batch_frame, return_tensors="pt").to(state.device)
                    if "dino" in args.model_path.lower():
                        frame_features = model(**inputs).last_hidden_state.mean(dim=1)
                    else:  # CLIP
                        frame_features = model.get_image_features(**inputs)

                    # Each video may have a different number of sampled frames.
                    # Map the flattened frame features back to their original shape.
                    batch_frame_features = torch.split(frame_features, frame_num_list)
                    batch_simi_cross_frame = [compute_adjacent_similarity(frame_features) for frame_features in batch_frame_features]
                    batch_similarity_mean = [
                        sum(simi_cross_frame) / len(simi_cross_frame) for simi_cross_frame in batch_simi_cross_frame
                    ]
                
                if args.video_folder == "":
                    saved_video_path_list = batch_video_path
                else:
                    saved_video_path_list = [os.path.relpath(video_path, args.video_folder) for video_path in batch_video_path]
                result_dict[args.video_path_column].extend(saved_video_path_list)
                result_dict["similarity_cross_frame"].extend(batch_simi_cross_frame)
                result_dict["similarity_mean"].extend(batch_similarity_mean)
                result_dict["sample_frame_idx"].extend(batch_sampled_frame_idx)

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
