import ast
from typing import Optional

import pandas as pd

from .logger import logger


# Ensure each item in the video_path_list matches the paths in the video_path column of the metadata.
def filter(
    video_path_list: list[str],
    basic_metadata_path: Optional[str] = None,
    min_resolution: float = 720*1280,
    min_duration: float = -1,
    max_duration: float = -1,
    aesthetic_score_metadata_path: Optional[str] = None,
    min_aesthetic_score: float = 4,
    aesthetic_score_siglip_metadata_path: Optional[str] = None,
    min_aesthetic_score_siglip: float = 4,
    text_score_metadata_path: Optional[str] = None,
    min_text_score: float = 0.02,
    motion_score_metadata_path: Optional[str] = None,
    min_motion_score: float = 2,
    max_motion_score: float = 999999,
    videoclipxl_score_metadata_path: Optional[str] = None,
    min_videoclipxl_score: float = 0.20,
    semantic_consistency_score_metadata_path: Optional[list[str]] = None,
    min_semantic_consistency_score: float = 0.80,
    video_path_column: str = "video_path"
):
    if basic_metadata_path is not None:
        if basic_metadata_path.endswith(".csv"):
            basic_df = pd.read_csv(basic_metadata_path)
        elif basic_metadata_path.endswith(".jsonl"):
            basic_df = pd.read_json(basic_metadata_path, lines=True)
        
        basic_df["resolution"] = basic_df["frame_size"].apply(lambda x: x[0] * x[1])
        filtered_basic_df = basic_df[basic_df["resolution"] < min_resolution]
        filtered_video_path_list = filtered_basic_df[video_path_column].tolist()

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        logger.info(
            f"Load {basic_metadata_path} ({len(basic_df)}) and filter {len(filtered_video_path_list)} videos "
            f"with resolution less than {min_resolution}."
        )

        if min_duration != -1:
            filtered_basic_df = basic_df[basic_df["duration"] < min_duration]
            filtered_video_path_list = filtered_basic_df[video_path_column].tolist()

            video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
            logger.info(
                f"Load {basic_metadata_path} and filter {len(filtered_video_path_list)} videos "
                f"with duration less than {min_duration}."
            )

        if max_duration != -1:
            filtered_basic_df = basic_df[basic_df["duration"] > max_duration]
            filtered_video_path_list = filtered_basic_df[video_path_column].tolist()

            video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
            logger.info(
                f"Load {basic_metadata_path} and filter {len(filtered_video_path_list)} videos "
                f"with duration greater than {max_duration}."
            )

    if aesthetic_score_metadata_path is not None:
        if aesthetic_score_metadata_path.endswith(".csv"):
            aesthetic_score_df = pd.read_csv(aesthetic_score_metadata_path)
        elif aesthetic_score_metadata_path.endswith(".jsonl"):
            aesthetic_score_df = pd.read_json(aesthetic_score_metadata_path, lines=True)

        # In pandas, csv will save lists as strings, whereas jsonl will not.
        aesthetic_score_df["aesthetic_score"] = aesthetic_score_df["aesthetic_score"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        aesthetic_score_df["aesthetic_score_mean"] = aesthetic_score_df["aesthetic_score"].apply(lambda x: sum(x) / len(x))
        filtered_aesthetic_score_df = aesthetic_score_df[aesthetic_score_df["aesthetic_score_mean"] < min_aesthetic_score]
        filtered_video_path_list = filtered_aesthetic_score_df[video_path_column].tolist()

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        logger.info(
            f"Load {aesthetic_score_metadata_path} ({len(aesthetic_score_df)}) and filter {len(filtered_video_path_list)} videos "
            f"with aesthetic score less than {min_aesthetic_score}."
        )

    if aesthetic_score_siglip_metadata_path is not None:
        if aesthetic_score_siglip_metadata_path.endswith(".csv"):
            aesthetic_score_siglip_df = pd.read_csv(aesthetic_score_siglip_metadata_path)
        elif aesthetic_score_siglip_metadata_path.endswith(".jsonl"):
            aesthetic_score_siglip_df = pd.read_json(aesthetic_score_siglip_metadata_path, lines=True)
        
        # In pandas, csv will save lists as strings, whereas jsonl will not.
        aesthetic_score_siglip_df["aesthetic_score_siglip"] = aesthetic_score_siglip_df["aesthetic_score_siglip"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        aesthetic_score_siglip_df["aesthetic_score_siglip_mean"] = aesthetic_score_siglip_df["aesthetic_score_siglip"].apply(
            lambda x: sum(x) / len(x)
        )
        filtered_aesthetic_score_siglip_df = aesthetic_score_siglip_df[
            aesthetic_score_siglip_df["aesthetic_score_siglip_mean"] < min_aesthetic_score_siglip
        ]
        filtered_video_path_list = filtered_aesthetic_score_siglip_df[video_path_column].tolist()

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        logger.info(
            f"Load {aesthetic_score_siglip_metadata_path} ({len(aesthetic_score_siglip_df)}) and filter {len(filtered_video_path_list)} videos "
            f"with aesthetic score (SigLIP) less than {min_aesthetic_score_siglip}."
        )

    if text_score_metadata_path is not None:
        if text_score_metadata_path.endswith(".csv"):
            text_score_df = pd.read_csv(text_score_metadata_path)
        elif text_score_metadata_path.endswith(".jsonl"):
            text_score_df = pd.read_json(text_score_metadata_path, lines=True)

        filtered_text_score_df = text_score_df[text_score_df["text_score"] > min_text_score]
        filtered_video_path_list = filtered_text_score_df[video_path_column].tolist()

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        logger.info(
            f"Load {text_score_metadata_path} ({len(text_score_df)}) and filter {len(filtered_video_path_list)} videos "
            f"with text score greater than {min_text_score}."
        )

    if motion_score_metadata_path is not None:
        if motion_score_metadata_path.endswith(".csv"):
            motion_score_df = pd.read_csv(motion_score_metadata_path)
        elif motion_score_metadata_path.endswith(".jsonl"):
            motion_score_df = pd.read_json(motion_score_metadata_path, lines=True)
        
        filtered_motion_score_df = motion_score_df[motion_score_df["motion_score"] < min_motion_score]
        filtered_video_path_list = filtered_motion_score_df[video_path_column].tolist()

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        logger.info(
            f"Load {motion_score_metadata_path} ({len(motion_score_df)}) and filter {len(filtered_video_path_list)} videos "
            f"with motion score smaller than {min_motion_score}."
        )

        filtered_motion_score_df = motion_score_df[motion_score_df["motion_score"] > max_motion_score]
        filtered_video_path_list = filtered_motion_score_df[video_path_column].tolist()

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        logger.info(
            f"Load {motion_score_metadata_path} ({len(motion_score_df)}) and filter {len(filtered_video_path_list)} videos "
            f"with motion score greater than {min_motion_score}."
        )
    
    if videoclipxl_score_metadata_path is not None:
        if videoclipxl_score_metadata_path.endswith(".csv"):
            videoclipxl_score_df = pd.read_csv(videoclipxl_score_metadata_path)
        elif videoclipxl_score_metadata_path.endswith(".jsonl"):
            videoclipxl_score_df = pd.read_json(videoclipxl_score_metadata_path, lines=True)
        
        filtered_videoclipxl_score_df = videoclipxl_score_df[videoclipxl_score_df["videoclipxl_score"] < min_videoclipxl_score]
        filtered_video_path_list = filtered_videoclipxl_score_df[video_path_column].tolist()

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        logger.info(
            f"Load {videoclipxl_score_metadata_path} ({len(videoclipxl_score_df)}) and "
            f"filter {len(filtered_video_path_list)} videos with mixclip score smaller than {min_videoclipxl_score}."
        )
    
    if semantic_consistency_score_metadata_path is not None:
        for f in semantic_consistency_score_metadata_path:
            if f.endswith(".csv"):
                semantic_consistency_score_df = pd.read_csv(f)
            elif f.endswith(".jsonl"):
                semantic_consistency_score_df = pd.read_json(f, lines=True)
            filtered_semantic_consistency_score_df = semantic_consistency_score_df[
                semantic_consistency_score_df["similarity_cross_frame"].apply(lambda x: min(x) < min_semantic_consistency_score)
            ]
            filtered_video_path_list = filtered_semantic_consistency_score_df[video_path_column].tolist()

            video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
            logger.info(
                f"Load {f} ({len(semantic_consistency_score_df)}) and filter {len(filtered_video_path_list)} videos "
                f"with the minimum semantic consistency score smaller than {min_semantic_consistency_score}."
            )
    
    return video_path_list
