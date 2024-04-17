import argparse
import os
import re
from tqdm import tqdm

import pandas as pd
from vllm import LLM, SamplingParams

from utils.logger import logger


def parse_args():
    parser = argparse.ArgumentParser(description="Recaption the video frame.")
    parser.add_argument(
        "--video_metadata_path", type=str, required=True, help="The path to the video dataset metadata (csv/jsonl)."
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
        default="sampled_frame_caption",
        help="The column contains the sampled_frame_caption.",
    )
    parser.add_argument(
        "--remove_quotes",
        action="store_true",
        help="Whether to remove quotes from caption.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        required=False,
        help="The batch size for the video caption.",
    )
    parser.add_argument(
        "--summary_model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
    )
    parser.add_argument(
        "--summary_prompt",
        type=str,
        default=(
            "You are a helpful video description generator. I'll give you a description of the middle frame of the video clip, "
            "which you need to summarize it into a description of the video clip."
            "Please provide your video description following these requirements: "
            "1. Describe the basic and necessary information of the video in the third person, be as concise as possible. "
            "2. Output the video description directly. Begin with 'In this video'. "
            "3. Limit the video description within 100 words. "
            "Here is the mid-frame description: "
        ),
    )
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--saved_freq", type=int, default=1000, help="The frequency to save the output results.")

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
    sampled_frame_caption_list = video_metadata_df[args.caption_column].tolist()
    
    if not (args.saved_path.endswith(".csv") or args.saved_path.endswith(".jsonl")):
        raise ValueError("The saved_path must end with .csv or .jsonl.")
    
    if os.path.exists(args.saved_path):
        if args.saved_path.endswith(".csv"):
            saved_metadata_df = pd.read_csv(args.saved_path)
        elif args.saved_path.endswith(".jsonl"):
            saved_metadata_df = pd.read_json(args.saved_path, lines=True)
        saved_video_path_list = saved_metadata_df[args.video_path_column].tolist()
        video_path_list = list(set(video_path_list) - set(saved_video_path_list))
        video_metadata_df.set_index(args.video_path_column, inplace=True)
        video_metadata_df = video_metadata_df.loc[video_path_list]
        sampled_frame_caption_list = video_metadata_df[args.caption_column].tolist()
        logger.info(f"Resume from {args.saved_path}: {len(saved_video_path_list)} processed and {len(video_path_list)} to be processed.")

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
    summary_model = LLM(model=args.summary_model_name, trust_remote_code=True)

    result_dict = {"video_path": [], "summary_model": [], "summary_caption": []}

    for i in tqdm(range(0, len(sampled_frame_caption_list), args.batch_size)):
        batch_video_path = video_path_list[i: i + args.batch_size]
        batch_caption = sampled_frame_caption_list[i : i + args.batch_size]
        batch_prompt = []
        for caption in batch_caption:
            if args.remove_quotes:
                caption = re.sub(r'(["\']).*?\1', "", caption)
            batch_prompt.append("user:" + args.summary_prompt + str(caption) + "\n assistant:")
        batch_output = summary_model.generate(batch_prompt, sampling_params)
        
        result_dict["video_path"].extend(batch_video_path)
        result_dict["summary_model"].extend([args.summary_model_name] * len(batch_caption))
        result_dict["summary_caption"].extend([output.outputs[0].text.rstrip() for output in batch_output])

        # Save the metadata every args.saved_freq.
        if i != 0 and ((i // args.batch_size) % args.saved_freq) == 0:
            result_df = pd.DataFrame(result_dict)
            if args.saved_path.endswith(".csv"):
                header = True if not os.path.exists(args.saved_path) else False
                result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
            elif args.saved_path.endswith(".jsonl"):
                result_df.to_json(args.saved_path, orient="records", lines=True, mode="a")
            logger.info(f"Save result to {args.saved_path}.")

            result_dict = {"video_path": [], "summary_model": [], "summary_caption": []}

    result_df = pd.DataFrame(result_dict)
    if args.saved_path.endswith(".csv"):
        header = True if not os.path.exists(args.saved_path) else False
        result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
    elif args.saved_path.endswith(".jsonl"):
        result_df.to_json(args.saved_path, orient="records", lines=True, mode="a")
    logger.info(f"Save the final result to {args.saved_path}.")


if __name__ == "__main__":
    main()