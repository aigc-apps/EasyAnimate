import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Convert jsonl to json.")
    parser.add_argument("--video_folder", type=str, default="", help="The video folder.")
    parser.add_argument(
        "--jsonl_load_path", type=str, default=None, help="The path to the video dataset metadata (csv/jsonl)."
    )
    parser.add_argument("--save_path", type=str, default=None, help="The save path to the output results.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.jsonl_load_path, "r") as read:
        _lines = read.readlines()

    output = []
    for line in _lines:
        try:
            line = json.loads(line.strip())
            videoid, name = line['video_path'], line['summary_caption']
            output.append(
                {
                    "file_path": os.path.join(args.video_folder, videoid),
                    "text": name,
                    "type": "video",
                }
            )
        except:
            pass

    with open(args.save_path, mode="w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()