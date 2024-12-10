import argparse
import json
import os
from multiprocessing import Manager, Pool
from pathlib import Path

import pandas as pd
from natsort import index_natsorted

from .get_meta_file import parallel_rglob
from .logger import logger


def process_file(file_path, shared_list):
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            shared_list.append(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Gather all jsonl files in a folder (meta_folder) to a single jsonl file (meta_file_path).")
    parser.add_argument("--meta_folder", type=str, required=True)
    parser.add_argument("--video_path_column", type=str, default="video_path")
    parser.add_argument("--meta_file_path", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--recursive", action="store_true", help="Whether to search sub-folders recursively.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.meta_folder):
        raise ValueError(f"The meta_folder {args.meta_folder} does not exist.")
    meta_folder = Path(args.meta_folder)
    if args.recursive:
        jsonl_files = [str(file) for file in parallel_rglob(meta_folder, f"*.jsonl", max_workers=args.n_jobs)]
    else:
        jsonl_files = [str(file) for file in meta_folder.glob(f"*.jsonl")]

    with Manager() as manager:
        shared_list = manager.list()
        with Pool(processes=args.n_jobs) as pool:
            for file_path in jsonl_files:
                pool.apply_async(process_file, args=(file_path, shared_list))
            pool.close()
            pool.join()

        with open(args.meta_file_path, "w") as f:
            for item in shared_list:
                f.write(json.dumps(item) + '\n')
    
    df = pd.read_json(args.meta_file_path, lines=True)
    df = df.iloc[index_natsorted(df[args.video_path_column])].reset_index(drop=True)
    logger.info(f"Save the gathered single jsonl file to {args.meta_file_path}.")
    df.to_json(args.meta_file_path, orient="records", lines=True, force_ascii=False)


if __name__ == '__main__':
    main()
