import argparse
import copy
import json
import os
import shutil
from multiprocessing import Pool

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from tqdm import tqdm

from utils.video_utils import download_video, get_video_path_list

tmp_file_dir = "./tmp"
DEFAULT_FFMPEG_ARGS = '-c:v libx264 -preset veryfast -crf 22 -c:a aac'

def parse_args():
    parser = argparse.ArgumentParser(
        description = '''Cut video by PySceneDetect''')
    parser.add_argument(
        'video',
        type = str, 
        help = '''Input format:
        1. Local video file path.
        2. Video URL.
        3. Local root dir path of videos.
        4. Local txt file of video urls/local file path, line by line.
        ''')
    parser.add_argument(
        '--threshold', 
        type = float,
        nargs='+', 
        default = [10, 20, 30],
        help = 'Threshold list the average change in pixel intensity must exceed to trigger a cut, one-to-one with frame_skip.')
    parser.add_argument(
        '--frame_skip', 
        type = int, 
        nargs='+',
        default = [0, 1, 2],
        help = 'Number list of frames to skip, coordinate with threshold \
        (i.e. process every 1 in N+1 frames, where N is frame_skip, \
        processing only 1/N+1 percent of the video, \
        speeding up the detection time at the expense of accuracy). One-to-one with threshold.')
    parser.add_argument(
        '--min_seconds', 
        type = int,
        default = 3,
        help = 'Video cut must be longer then min_seconds.')
    parser.add_argument(
        '--max_seconds', 
        type = int,
        default = 12,
        help = 'Video cut must be longer then min_seconds.')
    parser.add_argument(
        '--save_dir', 
        type = str, 
        default = "",
        help = 'Video scene cuts save dir, default value means reusing input video dir.')
    parser.add_argument(
        '--name_template', 
        type = str, 
        default = "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
        help = 'Video scene cuts save name template.')
    parser.add_argument(
        '--num_processes', 
        type = int, 
        default = os.cpu_count() // 2,
        help = 'Number of CPU cores to process the video scene cut.')
    parser.add_argument(
        "--save_json", action="store_true", help="Whether save json in datasets."
    )
    args = parser.parse_args()
    return args


def split_video_into_scenes(
    video_path: str, 
    threshold: list[float] = [27.0],
    frame_skip: list[int] = [0],
    min_seconds: int = 3,
    max_seconds: int = 8,
    save_dir: str = "",
    name_template: str = "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
    save_json: bool = False ):
    # SceneDetect video through casceded (threshold, FPS)
    frame_points = []
    frame_timecode = {}
    fps = 25.0
    for thre, f_skip in zip(threshold, frame_skip):
        # Open our video, create a scene manager, and add a detector.
        video = open_video(video_path, backend='pyav')
        scene_manager = SceneManager()
        scene_manager.add_detector(
            # [ContentDetector, ThresholdDetector, AdaptiveDetector]
            ContentDetector(threshold=thre, min_scene_len=10)
            )
        scene_manager.detect_scenes(video, frame_skip=f_skip, show_progress=False)
        scene_list = scene_manager.get_scene_list()
        for scene in scene_list:
            for frame_time_code in scene:
                frame_index = frame_time_code.get_frames()
                if frame_index not in frame_points:
                    frame_points.append(frame_index)
                frame_timecode[frame_index] = frame_time_code
                fps = frame_time_code.get_framerate()
        del video, scene_manager
    frame_points = sorted(frame_points)
    output_scene_list = []

    # Detect No Scene Change
    if len(frame_points) == 0:
        video = open_video(video_path, backend='pyav')
        frame_points = [0, video.duration.get_frames() - 1]
        frame_timecode = {
            frame_points[0]: video.base_timecode,
            frame_points[-1]: video.base_timecode + video.base_timecode + video.duration
        }
        del video

    for idx in range(len(frame_points) - 1):
        # Limit save out min seconds
        if frame_points[idx+1] - frame_points[idx] < fps * min_seconds:
            continue
        # Limit save out max seconds
        elif frame_points[idx+1] - frame_points[idx] > fps * max_seconds:
            tmp_start_timecode = copy.deepcopy(frame_timecode[frame_points[idx]])
            tmp_end_timecode = copy.deepcopy(frame_timecode[frame_points[idx]]) + int(max_seconds * fps)
            # Average cut by max seconds
            while tmp_end_timecode.get_frames() <= frame_points[idx+1]:
                output_scene_list.append((
                    copy.deepcopy(tmp_start_timecode), 
                    copy.deepcopy(tmp_end_timecode)))
                tmp_start_timecode += int(max_seconds * fps)
                tmp_end_timecode += int(max_seconds * fps)
            if tmp_end_timecode.get_frames() > frame_points[idx+1] and frame_points[idx+1] - tmp_start_timecode.get_frames() > fps * min_seconds:
                output_scene_list.append((
                    copy.deepcopy(tmp_start_timecode), 
                    frame_timecode[frame_points[idx+1]]))
            del tmp_start_timecode, tmp_end_timecode
            continue
        output_scene_list.append((
            frame_timecode[frame_points[idx]], 
            frame_timecode[frame_points[idx+1]]))
    
    # Reuse video dir
    if save_dir == "":
        save_dir = os.path.dirname(video_path)
    # Ensure save dir exists
    elif not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    clip_info_path = os.path.join(save_dir, os.path.splitext(os.path.basename(video_path))[0] + '.json')

    output_file_template = os.path.join(save_dir, name_template)
    split_video_ffmpeg(
        video_path, 
        output_scene_list, 
        arg_override=DEFAULT_FFMPEG_ARGS,
        output_file_template=output_file_template,
        show_progress=False, 
        show_output=False) # ffmpeg print
    
    if save_json:
        # Save clip info
        json.dump(
            [(frame_timecode_tuple[0].get_timecode(), frame_timecode_tuple[1].get_timecode()) for frame_timecode_tuple in output_scene_list], 
            open(clip_info_path, 'w'), 
            indent=2
        )

    return clip_info_path


def process_single_video(args):
    video, threshold, frame_skip, min_seconds, max_seconds, save_dir, name_template, save_json = args
    basename = os.path.splitext(os.path.basename(video))[0]
    # Video URL
    if video.startswith("http"):
        save_path = os.path.join(tmp_file_dir, f"{basename}.mp4")
        download_success = download_video(video, save_path)
        if not download_success:
            return
        video = save_path
    # Local video path
    else:
        if not os.path.isfile(video):
            print(f"Video not exists: {video}")
            return
    # SceneDetect video cut
    try:
        split_video_into_scenes(
            video_path=video,
            threshold=threshold,
            frame_skip=frame_skip,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            save_dir=save_dir,
            name_template=name_template,
            save_json=save_json
        )
    except Exception as e:
        print(e, video)


def main():
    # Args
    args = parse_args()
    video_input = args.video
    threshold = args.threshold
    frame_skip = args.frame_skip
    min_seconds = args.min_seconds
    max_seconds = args.max_seconds
    save_dir = args.save_dir
    name_template = args.name_template
    num_processes = args.num_processes
    save_json = args.save_json

    assert len(threshold) == len(frame_skip), \
        "Threshold must one-to-one match frame_skip."

    video_list = get_video_path_list(video_input)
    args_list = [
        (video, threshold, frame_skip, min_seconds, max_seconds, save_dir, name_template, save_json)
        for video in video_list
    ]

    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(video_list)) as progress_bar:
            for _ in pool.imap_unordered(process_single_video, args_list):
                progress_bar.update(1)


if __name__ == "__main__":
    main()
