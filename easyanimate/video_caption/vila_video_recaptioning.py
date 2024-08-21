# Modified from https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/vlm_demo_new.py.
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import load_checkpoint_and_dispatch, PartialState
from accelerate.utils import gather_object
from decord import VideoReader
from PIL import Image
from natsort import natsorted
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

import tinychat.utils.constants
# from tinychat.models.llava_llama import LlavaLlamaForCausalLM
from tinychat.models.vila_llama import VilaLlamaForCausalLM
from tinychat.stream_generators.llava_stream_gen import LlavaStreamGenerator
from tinychat.utils.conversation_utils import gen_params
from tinychat.utils.llava_image_processing import process_images
from tinychat.utils.prompt_templates import (
    get_image_token,
    get_prompter,
    get_stop_token_ids,
)
from tinychat.utils.tune import (
    device_warmup,
    tune_llava_patch_embedding,
)

from utils.filter import filter
from utils.logger import logger

gen_params.seed = 1
gen_params.temp = 1.0
gen_params.top_p = 1.0


def extract_uniform_frames(video_path: str, num_sampled_frames: int = 8):
    vr = VideoReader(video_path)
    sampled_frame_idx_list = np.linspace(0, len(vr), num_sampled_frames, endpoint=False, dtype=int)
    sampled_frame_list = []
    for idx in sampled_frame_idx_list:
        sampled_frame = Image.fromarray(vr[idx].asnumpy())
        sampled_frame_list.append(sampled_frame)

    return sampled_frame_list


def stream_output(output_stream):
    for outputs in output_stream:
        output_text = outputs["text"]
        output_text = output_text.strip().split(" ")
        # print(f"output_text: {output_text}.")
    return " ".join(output_text)


def skip(*args, **kwargs):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Recaption videos with VILA1.5.")
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
    parser.add_argument("--input_prompt", type=str, default="<video>\\n Elaborate on the visual and narrative elements of the video in detail.")
    parser.add_argument(
        "--model_type", type=str, default="LLaMa", help="type of the model"
    )
    parser.add_argument(
        "--model_path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8b-AWQ"
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--precision", type=str, default="W4A16", help="compute precision"
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
        default=100,
        help="The frequency to save the output results.",
    )

    parser.add_argument(
        "--basic_metadata_path", type=str, default=None, help="The path to the basic metadata (csv/jsonl)."
    )
    parser.add_argument("--min_resolution", type=float, default=0, help="The resolution threshold.")
    parser.add_argument("--min_duration", type=float, default=-1, help="The minimum duration.")
    parser.add_argument("--max_duration", type=float, default=-1, help="The maximum duration.")
    parser.add_argument(
        "--asethetic_score_metadata_path", type=str, default=None, help="The path to the video quality metadata (csv/jsonl)."
    )
    parser.add_argument("--min_asethetic_score", type=float, default=4.0, help="The asethetic score threshold.")
    parser.add_argument(
        "--asethetic_score_siglip_metadata_path", type=str, default=None, help="The path to the video quality metadata (csv/jsonl)."
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
    
    args = parser.parse_args()
    return args


def main(args):
    if args.video_metadata_path.endswith(".csv"):
        video_metadata_df = pd.read_csv(args.video_metadata_path)
    elif args.video_metadata_path.endswith(".jsonl"):
        video_metadata_df = pd.read_json(args.video_metadata_path, lines=True)
    else:
        raise ValueError("The video_metadata_path must end with .csv or .jsonl.")
    video_path_list = video_metadata_df[args.video_path_column].tolist()
    video_path_list = [os.path.basename(video_path) for video_path in video_path_list]

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
        asethetic_score_metadata_path=args.asethetic_score_metadata_path,
        min_asethetic_score=args.min_asethetic_score,
        asethetic_score_siglip_metadata_path=args.asethetic_score_siglip_metadata_path,
        min_asethetic_score_siglip=args.min_asethetic_score_siglip,
        text_score_metadata_path=args.text_score_metadata_path,
        min_text_score=args.min_text_score,
        motion_score_metadata_path=args.motion_score_metadata_path,
        min_motion_score=args.min_motion_score,
    )
    video_path_list = [os.path.join(args.video_folder, video_path) for video_path in video_path_list]
    # Sorting to guarantee the same result for each process.
    video_path_list = natsorted(video_path_list)

    state = PartialState()

    # Accelerate model initialization
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.model_path, "llm"), use_fast=False)
    tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX = (
        tokenizer.convert_tokens_to_ids(
            [tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
    )
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = VilaLlamaForCausalLM(config).half()
    tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN_IDX = (
        tokenizer.convert_tokens_to_ids(
            [tinychat.utils.constants.LLAVA_DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]
    )
    vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model()
    image_processor = vision_tower.image_processor
    # vision_tower = vision_tower.half()

    if args.precision == "W16A16":
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint shards")
        for i in pbar:
            model.llm = load_checkpoint_and_dispatch(
                model.llm,
                os.path.join(args.model_path, "llm"),
                no_split_module_classes=[
                    "OPTDecoderLayer",
                    "LlamaDecoderLayer",
                    "BloomBlock",
                    "MPTBlock",
                    "DecoderLayer",
                    "CLIPEncoderLayer",
                ],
            ).to(state.device)
        model = model.to(state.device)

    elif args.precision == "W4A16":
        from tinychat.utils.load_quant import load_awq_model
        # Auto load quant_path from the 3b/8b/13b/40b model.
        if args.quant_path is None:
            if "VILA1.5-3b-s2-AWQ" in args.model_path:
                args.quant_path = os.path.join(args.model_path, "llm/vila-1.5-3b-s2-w4-g128-awq-v2.pt")
            elif "VILA1.5-3b-AWQ" in args.model_path:
                args.quant_path = os.path.join(args.model_path, "llm/vila-1.5-3b-w4-g128-awq-v2.pt")
            elif "Llama-3-VILA1.5-8b-AWQ" in args.model_path:
                args.quant_path = os.path.join(args.model_path, "llm/llama-3-vila1.5-8b-w4-g128-awq-v2.pt")
            elif "VILA1.5-13b-AWQ" in args.model_path:
                args.quant_path = os.path.join(args.model_path, "llm/vila-1.5-13b-w4-g128-awq-v2.pt")
            elif "VILA1.5-40b-AWQ" in args.model_path:
                args.quant_path = os.path.join(args.model_path, "llm/vila-1.5-40b-w4-g128-awq-v2.pt")
        model.llm = load_awq_model(model.llm, args.quant_path, 4, 128, state.device)
        from tinychat.modules import (
            make_fused_mlp,
            make_fused_vision_attn,
            make_quant_attn,
            make_quant_norm,
        )

        make_quant_attn(model.llm, state.device)
        make_quant_norm(model.llm)
        # make_fused_mlp(model)
        # make_fused_vision_attn(model,state.device)
        model = model.to(state.device)

    else:
        raise NotImplementedError(f"Precision {args.precision} is not supported.")
    
    device_warmup(state.device)
    tune_llava_patch_embedding(vision_tower, device=state.device)

    stream_generator = LlavaStreamGenerator

    model_prompter = get_prompter(
        args.model_type, args.model_path, False, False
    )
    stop_token_ids = get_stop_token_ids(args.model_type, args.model_path)

    model.eval()

    index = len(video_path_list) - len(video_path_list) % state.num_processes
    # Avoid the NCCL timeout in the final gather operation.
    logger.info(f"Drop {len(video_path_list) % state.num_processes} videos to ensure each process handles the same number of videos.")
    video_path_list = video_path_list[:index]
    logger.info(f"{len(video_path_list)} videos are to be processed.")
    
    result_dict = {args.video_path_column: [], args.caption_column: []}
    with state.split_between_processes(video_path_list) as splitted_video_path_list:
        # TODO: Use VideoDataset.
        for i, video_path in enumerate(tqdm(splitted_video_path_list)):
            try:
                image_list = extract_uniform_frames(video_path, args.num_sampled_frames)
                image_num = len(image_list)
                # Similar operation in model_worker.py
                image_tensor = process_images(image_list, image_processor, model.config)
                if type(image_tensor) is list:
                    image_tensor = [
                        image.to(state.device, dtype=torch.float16) for image in image_tensor
                    ]
                else:
                    image_tensor = image_tensor.to(state.device, dtype=torch.float16)

                input_prompt = args.input_prompt
                # Insert image here
                image_token = get_image_token(model, args.model_path)
                image_token_holder = tinychat.utils.constants.LLAVA_DEFAULT_IM_TOKEN_PLACE_HOLDER
                im_token_count = input_prompt.count(image_token_holder)
                if im_token_count == 0:
                    model_prompter.insert_prompt(image_token * image_num + input_prompt)
                else:
                    assert im_token_count == image_num
                    input_prompt = input_prompt.replace(image_token_holder, image_token)
                    model_prompter.insert_prompt(input_prompt)
                output_stream = stream_generator(
                    model,
                    tokenizer,
                    model_prompter.model_input,
                    gen_params,
                    device=state.device,
                    stop_token_ids=stop_token_ids,
                    image_tensor=image_tensor,
                )
                outputs = stream_output(output_stream)
                if len(outputs) != 0:
                    result_dict[args.video_path_column].append(Path(video_path).name)
                    result_dict[args.caption_column].append(outputs)
            
            except Exception as e:
                logger.warning(f"VILA with {video_path} failed. Error is {e}.")

            if i != 0 and i % args.saved_freq == 0:
                state.wait_for_everyone()
                gathered_result_dict = {k: gather_object(v) for k, v in result_dict.items()}
                if state.is_main_process and len(gathered_result_dict[args.video_path_column]) != 0:
                    result_df = pd.DataFrame(gathered_result_dict)
                    if args.saved_path.endswith(".csv"):
                        header = False if os.path.exists(args.saved_path) else True
                        result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
                    elif args.saved_path.endswith(".jsonl"):
                        result_df.to_json(args.saved_path, orient="records", lines=True, mode="a", force_ascii=False)
                    logger.info(f"Save result to {args.saved_path}.")
                for k in result_dict.keys():
                    result_dict[k] = []
    
    state.wait_for_everyone()
    gathered_result_dict = {k: gather_object(v) for k, v in result_dict.items()}
    if state.is_main_process and len(gathered_result_dict[args.video_path_column]) != 0:
        result_df = pd.DataFrame(gathered_result_dict)
        if args.saved_path.endswith(".csv"):
            header = False if os.path.exists(args.saved_path) else True
            result_df.to_csv(args.saved_path, header=header, index=False, mode="a")
        elif args.saved_path.endswith(".jsonl"):
            result_df.to_json(args.saved_path, orient="records", lines=True, mode="a", force_ascii=False)
        logger.info(f"Save result to {args.saved_path}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
