import argparse
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from utils.logger import logger


MIN_SECONDS = int(os.getenv("MIN_SECONDS", 3))
MAX_SECONDS = int(os.getenv("MAX_SECONDS", 10))


def get_command(start_time, video_path, video_duration, output_path):
    # Use FFmpeg to split the video. Re-encoding is needed to ensure the accuracy of the clip
    # at the cost of consuming computational resources.
    return [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'panic',
        '-ss', str(start_time.time()),
        '-i', video_path,
        '-t', str(video_duration),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '22',
        '-c:a', 'aac',
        '-sn',
        output_path
    ]


def clip_video_star(args):
    return clip_video(*args)


def clip_video(video_path, timecode_list, output_folder, video_duration):
    """Recursively clip the video within the range of [MIN_SECONDS, MAX_SECONDS], 
    according to the timecode obtained from easyanimate/video_caption/cutscene_detect.py.
    """
    try:
        video_name = Path(video_path).stem

        if len(timecode_list) == 0:  # The video of a single scene.
            splitted_timecode_list = []
            start_time = datetime.strptime("00:00:00.000", "%H:%M:%S.%f")
            end_time = datetime.strptime(video_duration, "%H:%M:%S.%f")
            cur_start = start_time
            splitted_index = 0
            while cur_start < end_time:
                cur_end = min(cur_start + timedelta(seconds=MAX_SECONDS), end_time)
                cur_video_duration = (cur_end - cur_start).total_seconds()
                if cur_video_duration < MIN_SECONDS:
                    cur_start = cur_end
                    splitted_index += 1
                    continue
                splitted_timecode_list.append([cur_start.strftime("%H:%M:%S.%f")[:-3], cur_end.strftime("%H:%M:%S.%f")[:-3]])
                output_path = os.path.join(output_folder, video_name + f"_{splitted_index}.mp4")
                if os.path.exists(output_path):
                    logger.info(f"The clipped video {output_path} exists.")
                    cur_start = cur_end
                    splitted_index += 1
                    continue
                else:
                    command = get_command(cur_start, video_path, cur_video_duration, output_path)
                    try:
                        subprocess.run(command, check=True)
                    except Exception as e:
                        logger.warning(f"Run {command} error: {e}.")
                    finally:
                        cur_start = cur_end
                        splitted_index += 1

        for i, timecode in enumerate(timecode_list):  # The video of multiple scenes.
            start_time = datetime.strptime(timecode[0], "%H:%M:%S.%f")
            end_time = datetime.strptime(timecode[1], "%H:%M:%S.%f")
            video_duration = (end_time - start_time).total_seconds()
            output_path = os.path.join(output_folder, video_name + f"_{i}.mp4")
            if os.path.exists(output_path):
                logger.info(f"The clipped video {output_path} exists.")
                continue
            if video_duration < MIN_SECONDS:
                continue
            if video_duration > MAX_SECONDS:
                splitted_timecode_list = []
                cur_start = start_time
                splitted_index = 0
                while cur_start < end_time:
                    cur_end = min(cur_start + timedelta(seconds=MAX_SECONDS), end_time)
                    cur_video_duration = (cur_end - cur_start).total_seconds()
                    if cur_video_duration < MIN_SECONDS:
                        break
                    splitted_timecode_list.append([cur_start.strftime("%H:%M:%S.%f")[:-3], cur_end.strftime("%H:%M:%S.%f")[:-3]])
                    splitted_output_path = os.path.join(output_folder, video_name + f"_{i}_{splitted_index}.mp4")
                    if os.path.exists(splitted_output_path):
                        logger.info(f"The clipped video {splitted_output_path} exists.")
                        cur_start = cur_end
                        splitted_index += 1
                        continue
                    else:
                        command = get_command(cur_start, video_path, cur_video_duration, splitted_output_path)
                        try:
                            subprocess.run(command, check=True)
                        except Exception as e:
                            logger.warning(f"Run {command} error: {e}.")
                        finally:
                            cur_start = cur_end
                            splitted_index += 1
                
                continue
            
            # We found that the current scene detected by PySceneDetect includes a few frames from
            # the next scene occasionally. Directly discard the last few frames of the current scene.
            video_duration = video_duration - 0.5
            command = get_command(start_time, video_path, video_duration, output_path)
            subprocess.run(command, check=True)
    except Exception as e:
        logger.warning(f"Clip video with {video_path}. Error is: {e}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Splitting")
    parser.add_argument(
        "--video_metadata_path", type=str, default=None, help="The path to the video dataset metadata (csv/jsonl)."
    )
    parser.add_argument(
        "--video_path_column",
        type=str,
        default="video_path",
        help="The column contains the video path (an absolute path or a relative path w.r.t the video_folder).",
    )
    parser.add_argument("--video_folder", type=str, default="", help="The video folder.")
    parser.add_argument("--output_folder", type=str, default="outputs")
    parser.add_argument("--n_jobs", type=int, default=16)

    parser.add_argument("--resolution_threshold", type=float, default=0, help="The resolution threshold.")

    args = parser.parse_args()

    video_metadata_df = pd.read_json(args.video_metadata_path, lines=True)
    num_videos = len(video_metadata_df)
    video_metadata_df["resolution"] = video_metadata_df["frame_size"].apply(lambda x: x[0] * x[1])
    video_metadata_df = video_metadata_df[video_metadata_df["resolution"] >= args.resolution_threshold]
    logger.info(f"Filter {num_videos - len(video_metadata_df)} videos with resolution smaller than {args.resolution_threshold}.")
    video_path_list = video_metadata_df[args.video_path_column].to_list()
    video_id_list = [Path(video_path).stem for video_path in video_path_list]
    if len(video_id_list) != len(list(set(video_id_list))):
        logger.warning("Duplicate file names exist in the input video path list.")
    video_path_list = [os.path.join(args.video_folder, video_path) for video_path in video_path_list]
    video_timecode_list = video_metadata_df["timecode_list"].to_list()
    video_duration_list = video_metadata_df["duration"].to_list()

    assert len(video_path_list) == len(video_timecode_list)
    os.makedirs(args.output_folder, exist_ok=True)
    args_list = [
        (video_path, timecode_list, args.output_folder, video_duration)
        for video_path, timecode_list, video_duration in zip(
            video_path_list, video_timecode_list, video_duration_list
        )
    ]
    with Pool(args.n_jobs) as pool:
        # results = list(tqdm(pool.imap(clip_video_star, args_list), total=len(video_path_list)))
        results = pool.imap(clip_video_star, args_list)
        for result in tqdm(results, total=len(video_path_list)):
            pass