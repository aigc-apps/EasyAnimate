import argparse
import os

import pandas as pd
import torch
from natsort import natsorted
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams

from utils.filter import filter
from utils.logger import logger
from utils.video_dataset import VideoDataset, collate_fn


def recaption_batch_video(llm, batch_video_frames, prompt, sampling_params):
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": video_frames
            },
        }
        for video_frames in batch_video_frames
    ]

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    batch_output = []
    for o in outputs:
        generated_text = o.outputs[0].text
        batch_output.append(generated_text)

    return batch_output

def parse_args():
    parser = argparse.ArgumentParser(description="Recaption videos with InternVL2.")
    parser.add_argument(
        "--video_metadata_path",
        type=str,
        default=None,
        help="The path to the video dataset metadata (csv/jsonl).",
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
        default="caption",
        help="The column contains the caption.",
    )
    parser.add_argument(
        "--video_folder", type=str, default="", help="The video folder."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        required=False,
        help="The batch size for vllm inference. Adjust according to the number of GPUs to maximize inference throughput.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        required=False,
        help="The number of workers for the video dataset.",
    )
    parser.add_argument("--input_prompt", type=str, default="Describe this video in detail. Don\'t repeat.")
    parser.add_argument(
        "--model_path", type=str, default="OpenGVLab/InternVL2-40B-AWQ"
    )
    parser.add_argument(
        "--frame_sample_method",
        type=str,
        choices=["mid", "uniform", "image"],
        default="uniform",
    )
    parser.add_argument("--num_sampled_frames", type=int, default=8)
    parser.add_argument(
        "--saved_path",
        type=str,
        required=True,
        help="The save path to the output results (csv/jsonl).",
    )
    parser.add_argument(
        "--saved_freq",
        type=int,
        default=1,
        help="The frequency to save the output results.",
    )

    parser.add_argument(
        "--basic_metadata_path", type=str, default=None, help="The path to the basic metadata (csv/jsonl)."
    )
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
        logger.info(
            f"Resume from {args.saved_path}: {len(saved_video_path_list)} processed and {len(video_path_list)} to be processed."
        )
    
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
        semantic_consistency_score_metadata_path=args.semantic_consistency_score_metadata_path,
        min_semantic_consistency_score=args.min_semantic_consistency_score,
        video_path_column=args.video_path_column
    )
    # Sorting to guarantee the same result for each process.
    video_path_list = natsorted(video_path_list)

    video_dataset = VideoDataset(
        dataset_inputs={args.video_path_column: video_path_list},
        video_path_column=args.video_path_column,
        video_folder=args.video_folder,
        sample_method=args.frame_sample_method,
        num_sampled_frames=args.num_sampled_frames
    )
    video_loader = DataLoader(video_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    # Initialize the vllm inference pipeline.
    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", None)
    tensor_parallel_size = torch.cuda.device_count() if CUDA_VISIBLE_DEVICES is None else len(CUDA_VISIBLE_DEVICES.split(","))
    logger.info(f"Automatically set tensor_parallel_size={tensor_parallel_size} based on the available devices.")

    max_dynamic_patch = 1
    if args.frame_sample_method == "image":
        max_dynamic_patch = 12
    quantization = None
    if "awq" in args.model_path.lower():
        quantization="awq"
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=8192,
        limit_mm_per_prompt={"image": args.num_sampled_frames},
        gpu_memory_utilization=0.9,
        tensor_parallel_size=tensor_parallel_size,
        quantization=quantization,
        dtype="float16",
        mm_processor_kwargs={"max_dynamic_patch": max_dynamic_patch}
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.frame_sample_method == "image":
        placeholders = "<image>\n"
    else:
        placeholders = "".join(f"Frame{i}: <image>\n" for i in range(1, args.num_sampled_frames + 1))
    messages = [{"role": "user", "content": f"{placeholders}{args.input_prompt}"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    sampling_params = SamplingParams(temperature=0, max_tokens=512, stop_token_ids=stop_token_ids)

    result_dict = {args.video_path_column: [], args.caption_column: []}
    for idx, batch in enumerate(tqdm(video_loader)):
        if len(batch) > 0:
            batch_video_path = batch["path"]
            batch_frame = batch["sampled_frame"]  # [batch_size, num_sampled_frames, H, W, C]
            batch_caption = recaption_batch_video(llm, batch_frame, prompt, sampling_params)

            if args.video_folder == "":
                saved_video_path_list = batch_video_path
            else:
                saved_video_path_list = [os.path.relpath(video_path, args.video_folder) for video_path in batch_video_path]
            result_dict[args.video_path_column].extend(saved_video_path_list)
            result_dict["caption"].extend(batch_caption)
        
        if idx % args.saved_freq == 0 or idx == len(video_loader) - 1:
            result_df = pd.DataFrame(result_dict)

            # Append is not supported (oss).
            if args.saved_path.endswith(".csv"):
                if os.path.exists(args.saved_path):
                    saved_df = pd.read_csv(args.saved_path)
                    result_df = pd.concat([saved_df, result_df], ignore_index=True)
                result_df = result_df.iloc[natsorted(result_df.index, key=lambda x: result_df.loc[x, args.video_path_column])]
                result_df.to_csv(args.saved_path, index=False)
            elif args.saved_path.endswith(".jsonl"):
                if os.path.exists(args.saved_path):
                    saved_df = pd.read_json(args.saved_path, orient="records", lines=True)
                    result_df = pd.concat([saved_df, result_df], ignore_index=True)
                result_df = result_df.iloc[natsorted(result_df.index, key=lambda x: result_df.loc[x, args.video_path_column])]
                result_df.to_json(args.saved_path, orient="records", lines=True, force_ascii=False)
            logger.info(f"Save result to {args.saved_path}.")
            result_dict = {args.video_path_column: [], args.caption_column: []}
            

if __name__ == "__main__":
    main()