import argparse
import os

import easyocr
import numpy as np
import pandas as pd
from accelerate import PartialState
from accelerate.utils import gather_object
from natsort import natsorted
from tqdm import tqdm
from torchvision.datasets.utils import download_url

from utils.filter import filter
from utils.logger import logger
from utils.video_utils import extract_frames


def init_ocr_reader(root: str = "~/.cache/easyocr", device: str = "gpu"):
    root = os.path.expanduser(root)
    if not os.path.exists(root):
        os.makedirs(root)
    download_url(
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/easyanimate/video_caption/easyocr/craft_mlt_25k.pth",
        root,
        filename="craft_mlt_25k.pth",
        md5="2f8227d2def4037cdb3b34389dcf9ec1",
    )
    ocr_reader = easyocr.Reader(
        lang_list=["en", "ch_sim"],
        gpu=device,
        recognizer=False,
        verbose=False,
        model_storage_directory=root,
    )

    return ocr_reader


def triangle_area(p1, p2, p3):
    """Compute the triangle area according to its coordinates.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    tri_area = 0.5 * np.abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3)
    return tri_area


def compute_text_score(video_path, ocr_reader, sample_method="mid", num_sampled_frames=1):
    _, images = extract_frames(video_path, sample_method=sample_method, num_sampled_frames=num_sampled_frames)
    images = [np.array(image) for image in images]

    frame_ocr_area_ratios = []
    for image in images:
        # horizontal detected results and free-form detected
        horizontal_list, free_list = ocr_reader.detect(np.asarray(image))
        width, height = image.shape[0], image.shape[1]

        total_area = width * height
        # rectangles
        rect_area = 0
        for xmin, xmax, ymin, ymax in horizontal_list[0]:
            if xmax < xmin or ymax < ymin:
                continue
            rect_area += (xmax - xmin) * (ymax - ymin)
        # free-form
        quad_area = 0
        try:
            for points in free_list[0]:
                triangle1 = points[:3]
                quad_area += triangle_area(*triangle1)
                triangle2 = points[3:] + [points[0]]
                quad_area += triangle_area(*triangle2)
        except:
            quad_area = 0
        text_area = rect_area + quad_area

        frame_ocr_area_ratios.append(text_area / total_area)

    text_score = round(np.mean(frame_ocr_area_ratios), 5)

    return text_score


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the text score of the middle frame in the videos.")
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
        "--frame_sample_method",
        type=str,
        default="mid",
    )
    parser.add_argument(
        "--num_sampled_frames",
        type=int,
        default=1,
        help="num_sampled_frames",
    )
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
        motion_score_metadata_path=args.motion_score_metadata_path,
        min_motion_score=args.min_motion_score,
        max_motion_score=args.max_motion_score,
        semantic_consistency_score_metadata_path=args.semantic_consistency_score_metadata_path,
        min_semantic_consistency_score=args.min_semantic_consistency_score,
        video_path_column=args.video_path_column
    )
    video_path_list = [os.path.join(args.video_folder, video_path) for video_path in video_path_list]
    # Sorting to guarantee the same result for each process.
    video_path_list = natsorted(video_path_list)

    state = PartialState()
    if state.is_main_process:
        # Check if the model is downloaded in the main process.
        ocr_reader = init_ocr_reader(device="cpu")
    state.wait_for_everyone()
    ocr_reader = init_ocr_reader(device=state.device)

    index = len(video_path_list) - len(video_path_list) % state.num_processes
    # Avoid the NCCL timeout in the final gather operation.
    logger.info(
        f"Drop the last {len(video_path_list) % state.num_processes} videos to "
        "ensure each process handles the same number of videos."
    )
    video_path_list = video_path_list[:index]
    logger.info(f"{len(video_path_list)} videos are to be processed.")

    result_list = []
    with state.split_between_processes(video_path_list) as splitted_video_path_list:
        for i, video_path in enumerate(tqdm(splitted_video_path_list)):
            try:
                text_score = compute_text_score(
                    video_path,
                    ocr_reader,
                    sample_method=args.frame_sample_method,
                    num_sampled_frames=args.num_sampled_frames,
                )
                video_meta_info = {}
                if args.video_folder == "":
                    video_meta_info[args.video_path_column] = video_path
                else:
                    video_meta_info[args.video_path_column] = os.path.relpath(video_path, args.video_folder)
                video_meta_info["text_score"] = text_score
                result_list.append(video_meta_info)
            except Exception as e:
                logger.warning(f"Compute text score for video {video_path} with error: {e}.")
            if i % args.saved_freq == 0 or i == len(splitted_video_path_list) - 1:
                state.wait_for_everyone()
                gathered_result_list = gather_object(result_list)
                if state.is_main_process and len(gathered_result_list) != 0:
                    result_df = pd.DataFrame(gathered_result_list)
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
                result_list = []

if __name__ == "__main__":
    main()