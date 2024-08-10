import argparse
import copy
import os

import pandas as pd
from accelerate import PartialState
from accelerate.utils import gather_object
from natsort import natsorted
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.logger import logger
from utils.video_dataset import VideoDataset, collate_fn
from utils.video_utils import get_video_path_list, extract_frames


ACCELERATE_SUPPORTED_MODELS = ["Qwen-VL-Chat", "internlm-xcomposer2-vl-7b"]
SGLANG_SUPPORTED_MODELS = ["llava-v1.6-vicuna-7b"]


def parse_args():
    parser = argparse.ArgumentParser(description="Recaption the video frame.")
    parser.add_argument("--video_folder", type=str, default="", help="The video folder.")
    parser.add_argument(
        "--video_metadata_path", type=str, default=None, help="The path to the video dataset metadata (csv/jsonl/txt)."
    )
    parser.add_argument(
        "--video_path_column",
        type=str,
        default="video_path",
        help="The column contains the video path (an absolute path or a relative path w.r.t the video_folder).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        required=False,
        help="The batch size for the video dataset.",
    )
    parser.add_argument(
        "--frame_sample_method",
        type=str,
        choices=["mid", "uniform"],
        default="mid",
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=1,
        help="num_sampled_frames",
    )
    parser.add_argument(
        "--image_caption_model_name",
        type=str,
        choices=ACCELERATE_SUPPORTED_MODELS + SGLANG_SUPPORTED_MODELS,
        default="internlm-xcomposer2-vl-7b",
    )
    parser.add_argument(
        "--image_caption_model_quantized", type=bool, default=True, help="Whether to use the quantized image caption model."
    )
    parser.add_argument(
        "--image_caption_prompt",
        type=str,
        default="Describe this image and its style in a very detailed manner.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory to create the subfolder (named with the video name) to indicate the video has been processed.",
    )
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--saved_freq", type=int, default=1000, help="The frequency to save the output results.")

    args = parser.parse_args()
    return args


def accelerate_inference(args, video_path_list):
    from utils.image_captioner_awq import QwenVLChat, InternLMXComposer2

    state = PartialState()
    device = state.device
    if state.num_processes == 1:
        device = "cuda:0"
    if args.image_caption_model_name == "internlm-xcomposer2-vl-7b":
        image_caption_model = InternLMXComposer2(device=device, quantized=args.image_caption_model_quantized)
    elif args.image_caption_model_name == "Qwen-VL-Chat":
        image_caption_model = QwenVLChat(device=device, quantized=args.image_caption_model_quantized)
    
    # The workaround can be removed after https://github.com/huggingface/accelerate/pull/2781 is released.
    index = len(video_path_list) - len(video_path_list) % state.num_processes
    logger.info(f"Drop {len(video_path_list) % state.num_processes} videos to avoid duplicates in state.split_between_processes.")
    video_path_list = video_path_list[:index]
    
    if state.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    result_list = []
    with state.split_between_processes(video_path_list) as splitted_video_path_list:
        for i, video_path in enumerate(tqdm(splitted_video_path_list, desc=f"{state.device}")):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            try:
                if not os.path.exists(video_path):
                    print(f"Video {video_id} does not exist. Pass it.")
                    continue
                sampled_frame_list, sampled_frame_idx_list = extract_frames(video_path, num_sample_frames=args.num_sample_frames)
            except Exception as e:
                print(f"Failed to extract frames from video {video_id}. Error is {e}.")

            video_recaption_output_dir = os.path.join(args.output_dir, video_id)
            if os.path.exists(video_recaption_output_dir):
                print(f"Video {video_id} has been processed. Pass it.")
                continue
            else:
                os.makedirs(video_recaption_output_dir)

            caption_list = []
            for frame, frame_idx in zip(sampled_frame_list, sampled_frame_idx_list):
                frame_path = f"{args.output_dir}/{video_id}_{frame_idx}.png"
                frame.save(frame_path)
                try:
                    response, _ = image_caption_model(args.image_caption_prompt, frame_path)
                except Exception as e:
                    print(f"Failed to caption video {video_id}. Error is {e}.")
                finally:
                    os.remove(frame_path)
                caption_list.append(response)

            result_meta = {}
            if args.video_folder == "":
                result_meta[args.video_path_column] = video_path
            else:
                result_meta[args.video_path_column] = os.path.basename(video_path)
            result_meta["image_caption_model"] = args.image_caption_model_name
            result_meta["prompt"] = args.image_caption_prompt
            result_meta["sampled_frame_idx"] = sampled_frame_idx_list
            result_meta["sampled_frame_caption"] = caption_list
            result_list.append(copy.deepcopy(result_meta))

            # Save the metadata in the main process.
            if i != 0 and i % args.saved_freq == 0:
                state.wait_for_everyone()
                gathered_result_list = gather_object(result_list)
                if state.is_main_process:
                    result_df = pd.DataFrame(gathered_result_list)
                    if args.saved_path.endswith(".csv"):
                        result_df.to_csv(args.saved_path, index=False)
                    elif args.saved_path.endswith(".jsonl"):
                        result_df.to_json(args.saved_path, orient="records", lines=True)
                    print(f"Save result to {args.saved_path}.")

    # Wait for all processes to finish and gather the final result.
    state.wait_for_everyone()
    gathered_result_list = gather_object(result_list)
    # Save the metadata in the main process.
    if state.is_main_process:
        result_df = pd.DataFrame(gathered_result_list)
        if args.saved_path.endswith(".csv"):
            result_df.to_csv(args.saved_path, index=False)
        elif args.saved_path.endswith(".jsonl"):
            result_df.to_json(args.saved_path, orient="records", lines=True)
        print(f"Save the final result to {args.saved_path}.")


def sglang_inference(args, video_path_list):
    from utils.image_captioner_sglang import LLaVASRT

    if args.image_caption_model_name == "llava-v1.6-vicuna-7b":
        image_caption_model = LLaVASRT()
    
    result_dict = {
        "video_path": [],
        "image_caption_model": [],
        "prompt": [],
        'sampled_frame_idx': [],
        "sampled_frame_caption": []
    }

    video_dataset = VideoDataset(
        video_path_list=video_path_list,
        sample_method=args.frame_sample_method,
        num_sampled_frames=args.num_sampled_frames
    )
    video_loader = DataLoader(video_dataset, batch_size=args.batch_size, num_workers=16, collate_fn=collate_fn)
    for idx, batch in enumerate(tqdm(video_loader)):
        if len(batch) == 0:
            continue
        batch_video_path, batch_frame_idx = batch["video_path"], batch["sampled_frame_idx"]
        # [batch_size, num_sampled_frames, H, W, C] => [batch_size * num_sampled_frames, H, W, C].
        batch_frame = []
        for item_sampled_frame in batch["sampled_frame"]:
            batch_frame.extend([frame for frame in item_sampled_frame])

        try:
            response_list, _ = image_caption_model([args.image_caption_prompt] * len(batch_frame), batch_frame)
            response_list = [response_list[i:i + args.num_sampled_frames] for i in range(0, len(response_list), args.num_sampled_frames)]
        except Exception as e:
            logger.error(f"Failed to caption video {batch_video_path}. Error is {e}.")
        
        result_dict["video_path"].extend(batch_video_path)
        result_dict["image_caption_model"].extend([args.image_caption_model_name] * len(batch_video_path))
        result_dict["prompt"].extend([args.image_caption_prompt] * len(batch_video_path))
        result_dict["sampled_frame_idx"].extend(batch_frame_idx)
        result_dict["sampled_frame_caption"].extend(response_list)

        # Save the metadata in the main process.
        if idx != 0 and idx % args.saved_freq == 0:
            result_df = pd.DataFrame(result_dict)
            if args.saved_path.endswith(".csv"):
                header = True if not os.path.exists(args.saved_path) else False
                result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
            elif args.saved_path.endswith(".jsonl"):
                result_df.to_json(args.saved_path, orient="records", lines=True, mode="a")
            logger.info(f"Save result to {args.saved_path}.")

            result_dict = {
                "video_path": [],
                "image_caption_model": [],
                "prompt": [],
                'sampled_frame_idx': [],
                "sampled_frame_caption": []
            }

    if len(result_dict["video_path"]) != 0:
        result_df = pd.DataFrame(result_dict)
        if args.saved_path.endswith(".csv"):
            header = True if not os.path.exists(args.saved_path) else False
            result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
        elif args.saved_path.endswith(".jsonl"):
            result_df.to_json(args.saved_path, orient="records", lines=True, mode="a")
        logger.info(f"Save the final result to {args.saved_path}.")


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
        saved_video_path_list = [os.path.join(args.video_folder, path) for path in saved_video_path_list]
        video_path_list = list(set(video_path_list) - set(saved_video_path_list))
        # Sorting to guarantee the same result for each process.
        video_path_list = natsorted(video_path_list)
        logger.info(f"Resume from {args.saved_path}: {len(saved_video_path_list)} processed and {len(video_path_list)} to be processed.")
    
    if args.image_caption_model_name in SGLANG_SUPPORTED_MODELS:
        sglang_inference(args, video_path_list)
    elif args.image_caption_model_name in ACCELERATE_SUPPORTED_MODELS:
        accelerate_inference(args, video_path_list)
    else:
        raise ValueError(f"The {args.image_caption_model_name} is not supported.")


if __name__ == "__main__":
    main()
