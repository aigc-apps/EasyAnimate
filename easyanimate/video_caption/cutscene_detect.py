import argparse
import os
from copy import deepcopy
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from tqdm import tqdm

from utils.logger import logger


def cutscene_detection_star(args):
    return cutscene_detection(*args)


def cutscene_detection(video_path, saved_path, cutscene_threshold=27, min_scene_len=15):
    try:
        if os.path.exists(saved_path):
            logger.info(f"{video_path} has been processed.")
            return
        # Use PyAV as the backend to avoid (to some exent) containing the last frame of the previous scene.
        # https://github.com/Breakthrough/PySceneDetect/issues/279#issuecomment-2152596761.
        video = open_video(video_path, backend="pyav")
        frame_rate, frame_size = video.frame_rate, video.frame_size
        duration = deepcopy(video.duration)

        frame_points, frame_timecode = [], {}
        scene_manager = SceneManager()
        scene_manager.add_detector(
            # [ContentDetector, ThresholdDetector, AdaptiveDetector]
            ContentDetector(threshold=cutscene_threshold, min_scene_len=min_scene_len)
        )
        scene_manager.detect_scenes(video, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        for scene in scene_list:
            for frame_time_code in scene:
                frame_index = frame_time_code.get_frames()
                if frame_index not in frame_points:
                    frame_points.append(frame_index)
                frame_timecode[frame_index] = frame_time_code
        
        del video, scene_manager
        
        frame_points = sorted(frame_points)
        output_scene_list = []
        for idx in range(len(frame_points) - 1):
            output_scene_list.append((frame_timecode[frame_points[idx]], frame_timecode[frame_points[idx+1]]))
        
        timecode_list = [(frame_timecode_tuple[0].get_timecode(), frame_timecode_tuple[1].get_timecode()) for frame_timecode_tuple in output_scene_list]
        meta_scene = [{
            "video_path": Path(video_path).name,
            "timecode_list": timecode_list,
            "fram_rate": frame_rate,
            "frame_size": frame_size,
            "duration": str(duration)  # __repr__
        }]
        pd.DataFrame(meta_scene).to_json(saved_path, orient="records", lines=True)
    except Exception as e:
        logger.warning(f"Cutscene detection with {video_path} failed. Error is: {e}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cutscene Detection")
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
    parser.add_argument("--saved_folder", type=str, required=True, help="The save path to the output results (csv/jsonl).")
    parser.add_argument("--n_jobs", type=int, default=1, help="The number of processes.")

    args = parser.parse_args()

    metadata_df = pd.read_json(args.video_metadata_path, lines=True)
    video_path_list = metadata_df[args.video_path_column].tolist()
    video_path_list = [os.path.join(args.video_folder, video_path) for video_path in video_path_list]

    if not os.path.exists(args.saved_folder):
        os.makedirs(args.saved_folder, exist_ok=True)
    # The glob can be slow when there are many small jsonl files.
    saved_path_list = [os.path.join(args.saved_folder, Path(video_path).stem + ".jsonl") for video_path in video_path_list]
    args_list = [
        (video_path, saved_path)
        for video_path, saved_path in zip(video_path_list, saved_path_list)
    ]
    # Since the length of the video is not uniform, the gather operation is not performed.
    # We need to run easyanimate/video_caption/utils/gather_jsonl.py after the program finised.
    with Pool(args.n_jobs) as pool:
        results = list(tqdm(pool.imap(cutscene_detection_star, args_list), total=len(video_path_list)))
