import ast
import os

import pandas as pd


def filter(
    video_path_list,
    basic_metadata_path=None,
    min_resolution=720*1280,
    asethetic_score_metadata_path=None,
    min_asethetic_score=4,
    text_score_metadata_path=None,
    min_text_score=0.02,
    motion_score_metadata_path=None,
    min_motion_score=2,
    video_path_column="video_path"
):
    video_path_list = [os.path.basename(video_path) for video_path in video_path_list]

    if basic_metadata_path is not None:
        if basic_metadata_path.endswith(".csv"):
            basic_df = pd.read_csv(basic_metadata_path)
        elif basic_metadata_path.endswith(".jsonl"):
            basic_df = pd.read_json(basic_metadata_path, lines=True)
        
        basic_df["resolution"] = basic_df["frame_size"].apply(lambda x: x[0] * x[1])
        filtered_basic_df = basic_df[basic_df["resolution"] < min_resolution]
        filtered_video_path_list = filtered_basic_df[video_path_column].tolist()
        filtered_video_path_list = [os.path.basename(video_path) for video_path in filtered_video_path_list]

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        print(f"Load {basic_metadata_path} and filter {len(filtered_video_path_list)} videos with resolution less than {min_resolution}.")
    
    if asethetic_score_metadata_path is not None:
        if asethetic_score_metadata_path.endswith(".csv"):
            asethetic_score_df = pd.read_csv(asethetic_score_metadata_path)
        elif asethetic_score_metadata_path.endswith(".jsonl"):
            asethetic_score_df = pd.read_json(asethetic_score_metadata_path, lines=True)

        # In pandas, csv will save lists as strings, whereas jsonl will not.
        asethetic_score_df["aesthetic_score"] = asethetic_score_df["aesthetic_score"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        asethetic_score_df["aesthetic_score_mean"] = asethetic_score_df["aesthetic_score"].apply(lambda x: sum(x) / len(x))
        filtered_asethetic_score_df = asethetic_score_df[asethetic_score_df["aesthetic_score_mean"] < min_asethetic_score]
        filtered_video_path_list = filtered_asethetic_score_df[video_path_column].tolist()
        filtered_video_path_list = [os.path.basename(video_path) for video_path in filtered_video_path_list]

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        print(f"Load {asethetic_score_metadata_path} and filter {len(filtered_video_path_list)} videos with aesthetic score less than {min_asethetic_score}.")

    if text_score_metadata_path is not None:
        if text_score_metadata_path.endswith(".csv"):
            text_score_df = pd.read_csv(text_score_metadata_path)
        elif text_score_metadata_path.endswith(".jsonl"):
            text_score_df = pd.read_json(text_score_metadata_path, lines=True)

        filtered_text_score_df = text_score_df[text_score_df["text_score"] > min_text_score]
        filtered_video_path_list = filtered_text_score_df[video_path_column].tolist()
        filtered_video_path_list = [os.path.basename(video_path) for video_path in filtered_video_path_list]

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        print(f"Load {text_score_metadata_path} and filter {len(filtered_video_path_list)} videos with text score greater than {min_text_score}.")
    
    if motion_score_metadata_path is not None:
        if motion_score_metadata_path.endswith(".csv"):
            motion_score_df = pd.read_csv(motion_score_metadata_path)
        elif motion_score_metadata_path.endswith(".jsonl"):
            motion_score_df = pd.read_json(motion_score_metadata_path, lines=True)
        
        filtered_motion_score_df = motion_score_df[motion_score_df["motion_score"] < min_motion_score]
        filtered_video_path_list = filtered_motion_score_df[video_path_column].tolist()
        filtered_video_path_list = [os.path.basename(video_path) for video_path in filtered_video_path_list]

        video_path_list = list(set(video_path_list).difference(set(filtered_video_path_list)))
        print(f"Load {motion_score_metadata_path} and filter {len(filtered_video_path_list)} videos with motion score smaller than {min_motion_score}.")
    
    return video_path_list