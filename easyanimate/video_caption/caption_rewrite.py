import argparse
import os
import re
from copy import deepcopy

import pandas as pd
import torch
from natsort import index_natsorted
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.logger import logger


def extract_output(s, prefix='"rewritten description": '):
    """Customize the function according to the prompt."""
    # Since some LLMs struggles to output strictly formatted JSON strings as specified by the prompt,
    # thus manually parse the output string `{"rewritten description": "your rewritten description here"}`.
    match = re.search(r"{(.+?)}", s, re.DOTALL)
    if not match:
        logger.warning(f"{s} is not in the json format. Return None.")
        return None
    output = match.group(1).strip()
    if output.startswith(prefix):
        output = output[len(prefix) :]
        if output[0] == '"' and output[-1] == '"':
            return output[1:-1]
        else:
            logger.warning(f"{output} does not start and end with the double quote. Return None.")
            return None
    else:
        logger.warning(f"{output} does not start with {prefix}. Return None.")
        return None

"""The file unifies the following two tasks:
1. Caption Rewrite: rewrite the video recaption results by LLMs.
2. Beautiful Prompt: rewrite and beautify the user-uploaded prompt via LLMs.

For the caption rewrite task, the input video_metadata_path should have the following format:
```jsonl
{"video_path_column": "1.mp4", "caption_column": "a man is running in the street."}
...
{"video_path_column": "100.mp4", "caption_column": "a dog is chasing a cat."}
```
The video_path_column in the argparse must be specified.

For the beautiful prompt task, the input video_metadata_path should have the following format:
```jsonl
{"caption_column": "a man is running in the street."}
...
{"caption_column": "a dog is chasing a cat."}
```
The beautiful_prompt_column in the argparse must be specified for the saving purpose.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite the video caption by LLMs.")
    parser.add_argument(
        "--video_metadata_path", type=str, required=True, help="The path to the video dataset metadata (csv/jsonl)."
    )
    parser.add_argument(
        "--video_path_column",
        type=str,
        default=None,
        help=(
            "The column contains the video path (an absolute path or a relative path w.r.t the video_folder)."
            "It is conflicted with the beautiful_prompt_column."
        ),
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column contains the video caption.",
    )
    parser.add_argument(
        "--beautiful_prompt_column",
        type=str,
        default=None,
        help="The column name for the beautiful prompt column. It is conflicted with the video_path_column.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        required=False,
        help="The batch size for vllm inference. Adjust according to the number of GPUs to maximize inference throughput.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="NousResearch/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="A string or a txt file contains the prompt.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="The prefix to extract the output from LLMs.",
    )
    parser.add_argument(
        "--answer_template",
        type=str,
        default="",
        help="The anwer template in the prompt. If specified, rewritten results same as the answer template will be removed.",
    )
    parser.add_argument(
        "--max_retry_count",
        type=int,
        default=1,
        help="The maximum retry count to ensure outputs with the valid format from LLMs.",
    )
    parser.add_argument("--saved_path", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--saved_freq", type=int, default=1, help="The frequency to save the output results.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.video_metadata_path.endswith(".csv"):
        video_metadata_df = pd.read_csv(args.video_metadata_path)
    elif args.video_metadata_path.endswith(".jsonl"):
        video_metadata_df = pd.read_json(args.video_metadata_path, lines=True)
    elif args.video_metadata_path.endswith(".json"):
        video_metadata_df = pd.read_json(args.video_metadata_path)
    else:
        raise ValueError(f"The {args.video_metadata_path} must end with .csv, .jsonl or .json.")

    saved_suffix = os.path.splitext(args.saved_path)[1]
    if saved_suffix not in set([".csv", ".jsonl", ".json"]):
        raise ValueError(f"The saved_path must end with .csv, .jsonl or .json.")
    
    if args.video_path_column is None and args.beautiful_prompt_column is None:
        raise ValueError("Either video_path_column or beautiful_prompt_column should be specified in the arguments.")
    if args.video_path_column is not None and args.beautiful_prompt_column is not None:
        raise ValueError(
            "Both video_path_column and beautiful_prompt_column can not be specified in the arguments at the same time."
        )

    if os.path.exists(args.saved_path):
        if args.saved_path.endswith(".csv"):
            saved_metadata_df = pd.read_csv(args.saved_path)
        elif args.saved_path.endswith(".jsonl"):
            saved_metadata_df = pd.read_json(args.saved_path, lines=True)

        if args.video_path_column is not None:
            # Filter out the unprocessed video-caption pairs by setting the indicator=True.
            merged_df = video_metadata_df.merge(saved_metadata_df, on=args.video_path_column, how="outer", indicator=True)
            video_metadata_df = merged_df[merged_df["_merge"] == "left_only"]
            # Sorting to guarantee the same result for each process.
            video_metadata_df = video_metadata_df.iloc[index_natsorted(video_metadata_df[args.video_path_column])]
            video_metadata_df = video_metadata_df.reset_index(drop=True)
        if args.beautiful_prompt_column is not None:
            # Filter out the unprocessed caption-beautifil_prompt pairs by setting the indicator=True.
            merged_df = video_metadata_df.merge(saved_metadata_df, on=args.caption_column, how="outer", indicator=True)
            video_metadata_df = merged_df[merged_df["_merge"] == "left_only"]
            # Sorting to guarantee the same result for each process.
            video_metadata_df = video_metadata_df.iloc[index_natsorted(video_metadata_df[args.caption_column])]
            video_metadata_df = video_metadata_df.reset_index(drop=True)
        logger.info(
            f"Resume from {args.saved_path}: {len(saved_metadata_df)} processed and {len(video_metadata_df)} to be processed."
        )

    if args.prompt.endswith(".txt") and os.path.exists(args.prompt):
        with open(args.prompt, "r") as f:
            args.prompt = "".join(f.readlines())
    logger.info(f"Prompt: {args.prompt}")

    if args.max_retry_count < 1:
        raise ValueError(f"The max_retry_count {args.max_retry_count} must be greater than 0.")

    if args.video_path_column is not None:
        video_path_list = video_metadata_df[args.video_path_column].tolist()
    if args.caption_column in video_metadata_df.columns:
        sampled_frame_caption_list = video_metadata_df[args.caption_column].tolist()
    else:
        # When two columns with the same name, the dataframe merge operation on will distinguish them by adding 'x' and 'y'.
        sampled_frame_caption_list = video_metadata_df[args.caption_column + "_x"].tolist()

    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", None)
    tensor_parallel_size = torch.cuda.device_count() if CUDA_VISIBLE_DEVICES is None else len(CUDA_VISIBLE_DEVICES.split(","))
    logger.info(f"Automatically set tensor_parallel_size={tensor_parallel_size} based on the available devices.")

    llm = LLM(model=args.model_name, trust_remote_code=True, tensor_parallel_size=tensor_parallel_size)
    if "Meta-Llama-3" in args.model_name:
        if "Meta-Llama-3-70B" in args.model_name:
            # Llama-3-70B should use the tokenizer from Llama-3-8B
            # https://github.com/vllm-project/vllm/issues/4180#issuecomment-2068292942
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        sampling_params = SamplingParams(temperature=0.7, top_p=1, max_tokens=1024, stop_token_ids=stop_token_ids)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        sampling_params = SamplingParams(temperature=0.7, top_p=1, max_tokens=1024)

    if args.video_path_column is not None:
        result_dict = {args.video_path_column: [], args.caption_column: []}
    if args.beautiful_prompt_column is not None:
        result_dict = {args.caption_column: [], args.beautiful_prompt_column: []}

    for i in tqdm(range(0, len(sampled_frame_caption_list), args.batch_size)):
        if args.video_path_column is not None:
            batch_video_path = video_path_list[i : i + args.batch_size]
        batch_caption = sampled_frame_caption_list[i : i + args.batch_size]
        batch_prompt = []
        for caption in batch_caption:
            # batch_prompt.append("user:" + args.prompt + str(caption) + "\n assistant:")
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt + "\n" + str(caption)},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_prompt.append(text)
        
        cur_retry_count = 0
        while cur_retry_count < args.max_retry_count:
            if len(batch_prompt) == 0:
                break

            batch_result = []
            batch_output = llm.generate(batch_prompt, sampling_params)
            batch_output = [output.outputs[0].text.rstrip() for output in batch_output]
            if args.prefix is not None:
                batch_output = [extract_output(output, args.prefix) for output in batch_output]

            if args.video_path_column is not None:
                retry_batch_video_path, retry_batch_prompt = [], []
                for (video_path, prompt, output) in zip(batch_video_path, batch_prompt, batch_output):
                    # Filter out data that does not meet the output format to retry.
                    if output is not None and output != args.answer_template:
                        batch_result.append((video_path, output))
                    else:
                        retry_batch_video_path.append(video_path)
                        retry_batch_prompt.append(prompt)
                if len(batch_result) != 0:
                    batch_video_path, batch_output = zip(*batch_result)
                    result_dict[args.video_path_column].extend(deepcopy(batch_video_path))
                    result_dict[args.caption_column].extend(deepcopy(batch_output))
                
                batch_video_path, batch_prompt = retry_batch_video_path, retry_batch_prompt
            if args.beautiful_prompt_column is not None:
                retry_batch_caption, retry_batch_prompt = [], []
                for (caption, prompt, output) in zip(batch_caption, batch_prompt, batch_output):
                    # Filter out data that does not meet the output format to retry.
                    if output is not None and output != args.answer_template:
                        batch_result.append((caption, output))
                    else:
                        retry_batch_caption.append(caption)
                        retry_batch_prompt.append(prompt)
                if len(batch_result) != 0:
                    batch_caption, batch_output = zip(*batch_result)
                    result_dict[args.caption_column].extend(deepcopy(batch_caption))
                    result_dict[args.beautiful_prompt_column].extend(deepcopy(batch_output))
                
                batch_caption, batch_prompt = retry_batch_caption, retry_batch_prompt
            
            cur_retry_count += 1
            logger.info(
                f"Current retry count/Maximum retry count: {cur_retry_count}/{args.max_retry_count}.: "
                f"Retrying {len(batch_prompt)} prompts with invalid output format."
            )

        # Save the metadata every args.saved_freq.
        if (i // args.batch_size) % args.saved_freq == 0 or (i + 1) * args.batch_size >= len(sampled_frame_caption_list):
            if len(result_dict[args.caption_column]) > 0:
                result_df = pd.DataFrame(result_dict)
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

            result_dict = {args.caption_column: []}
            if args.video_path_column is not None:
                result_dict = {args.video_path_column: [], args.caption_column: []}
            if args.beautiful_prompt_column is not None:
                result_dict = {args.caption_column: [], args.beautiful_prompt_column: []}

if __name__ == "__main__":
    main()
