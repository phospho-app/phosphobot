#!/usr/bin/env python3

"""
This script will process all parquets files in the specified folder by correcting the episode_index column.
We then compute the statistics using the lerobot_stats_compute.py script.
"""

import os
import re
import sys
import glob
import tyro
import json
import subprocess
import pandas as pd
from dataclasses import dataclass


@dataclass
class Config:
    """
    If no indexes_to_delete is provided, will attempt to repair the dataset.
    If indexes_to_delete is provided, will delete the specified indexes from the dataset and repair the rest.
    """

    dataset_path: str
    """the path to the dataset to repair"""
    indexes_to_delete: str | None = None
    """the indexes to delete, comma separated"""


def check_v2(info_path):
    if not os.path.exists(info_path):
        print(f"Error: {info_path} does not exist")
        sys.exit(1)
    with open(info_path, "r") as f:
        try:
            info = json.load(f)
            if "codebase_version" not in info:
                print(f"Error: {info_path} is not a valid v2.0 dataset")
                sys.exit(1)
            elif info["codebase_version"] != "v2.0":
                print(
                    f"Error: {info_path} is not a v2.0 dataset, found {info['codebase_version']}"
                )
                sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: {info_path} is not a valid JSON file")
            sys.exit(1)


def delete_DS_Store(dataset_path):
    """
    Delete all .DS_Store files in the given dataset path and its subdirectories.
    """
    print("Deleting .DS_Store files...")
    ds_store_files = glob.glob(
        os.path.join(dataset_path, "**", ".DS_Store"), recursive=True
    )

    if not ds_store_files:
        print("No .DS_Store files found")
        return

    for file in ds_store_files:
        os.remove(file)
        print(f"Deleted {file}")

    print(".DS_Store files deleted")


def process_parquet_files(folder_path):
    """
    Process all parquet files in the given folder by correcting the episode_index column.
    The value in episode_index will match the episode number in the filename.

    Args:
        folder_path (str): Path to the folder containing parquet files
    """
    print("Processing parquet files...")
    parquet_files = glob.glob(os.path.join(folder_path, "episode_*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {folder_path}")
        return

    print(f"Found {len(parquet_files)} parquet files to process")

    # Order the files by episode number in ascending order
    parquet_files.sort(
        key=lambda x: int(re.search(r"episode_(\d+)\.parquet", x).group(1))
    )

    # Check if the episode number is continuous, if not, rename the parquet files and the corresponding videos
    episode_numbers = [
        int(re.search(r"episode_(\d+)\.parquet", file).group(1))
        for file in parquet_files
    ]

    # Make sure the list is ordered
    episode_numbers.sort()

    # Get names of the video folders in video_path
    video_folder = os.listdir(videos_folder_path)

    if episode_numbers != list(range(len(episode_numbers))):
        print(
            "Episode numbers are not continuous or starting from 0. Renaming files and videos..."
        )
        for i, file in enumerate(parquet_files):
            # We always start from 0
            new_episode_number = i
            new_file = os.path.join(
                folder_path, f"episode_{new_episode_number:06d}.parquet"
            )
            os.rename(file, new_file)
            print(f"Renamed {file} to {new_file}")

            # Rename the corresponding video files
            for folder in video_folder:
                new_video_file = os.path.join(
                    videos_folder_path, folder, f"episode_{new_episode_number:06d}.mp4"
                )
                video_file = os.path.join(
                    videos_folder_path, folder, f"episode_{episode_numbers[i]:06d}.mp4"
                )
                os.rename(video_file, new_video_file)
                print(f"Renamed {video_file} to {new_video_file}")

        # Update the list of parquet files after renaming
        parquet_files = glob.glob(os.path.join(folder_path, "episode_*.parquet"))
        parquet_files.sort(
            key=lambda x: int(re.search(r"episode_(\d+)\.parquet", x).group(1))
        )
        print("Updated parquet files list after renaming")

    # Process each parquet file
    total_index = 0
    for file_path in parquet_files:
        # Extract episode number from filename using regex
        filename = os.path.basename(file_path)
        match = re.search(r"episode_(\d+)\.parquet", filename)

        if match:
            episode_number = int(match.group(1))
            print(f"Processing {filename} - Episode {episode_number}")

            try:
                # Read the parquet file
                df = pd.read_parquet(file_path, engine="pyarrow")

                # Add episode_index column with the extracted number
                df["episode_index"] = episode_number

                # Rewrite frame_index column to go from 0 to n-1
                df["frame_index"] = range(len(df))

                # Rewrite index column to be a rolling index
                df["index"] = range(total_index, total_index + len(df))
                total_index += len(df)

                # Save the modified DataFrame back to the same file
                df.to_parquet(file_path, index=False)

                print(f"Successfully updated {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                sys.exit(1)
        else:
            print(f"Skipping {filename} - doesn't match expected pattern")

    print("Parquet processing complete")


def run_stats_script(dataset_path):
    """Run the lerobot_stats_compute.py script with uv, fallback to python"""
    script_path = "lerobot_stats_compute.py"

    print("Running lerobot_stats_compute.py...")

    try:
        subprocess.run(
            ["uv", "run", script_path, "--dataset-path", dataset_path],
            check=True,
        )
        print(f"Successfully executed {script_path} with uv")

    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_path}: {str(e)}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Execution error: {str(e)}")
        sys.exit(1)


def delete_indexes(indexes: list[int]):
    # Delete the parquet files
    print("Deleting parquet files...")
    parquet_files = glob.glob(os.path.join(parquets_folder_path, "*.parquet"))
    for index in indexes:
        for file in parquet_files:
            if f"episode_{index:06d}.parquet" in file:
                os.remove(file)
                print(f"Deleted file {file}")

    # Delete the corresponding video files
    print("Deleting video files...")
    video_folders = os.listdir(videos_folder_path)
    for index in indexes:
        for folder in video_folders:
            video_files = glob.glob(
                os.path.join(videos_folder_path, folder, f"episode_{index:06d}.mp4")
            )
            for video_file in video_files:
                os.remove(video_file)
                print(f"Deleted file {video_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: uv run repair_dataset.py --dataset-path <dataset_path> --indexes-to-delete <indexes_to_delete_comma_separated>\nFor example: python repair_dataset.py --dataset-path /path/to/dataset --indexes-to-delete 2,7,12"
        )
        sys.exit(1)

    # Parse arguments using tyro
    config = tyro.cli(Config)

    # dataset_path is the parent folder of the parquet files
    dataset_path = config.dataset_path
    parquets_folder_path = os.path.join(dataset_path, "data", "chunk-000")
    videos_folder_path = os.path.join(dataset_path, "videos", "chunk-000")
    info_file_path = os.path.join(dataset_path, "meta", "info.json")

    # Check that the dataset is in the v2.0 format
    check_v2(info_file_path)

    # Add message and ask the user to press enter to continue
    message = ""
    if config.indexes_to_delete and config.dataset_path:
        message = f"This script will delete the following episodes: {config.indexes_to_delete} and repair the rest of your dataset: {config.dataset_path}.\nPress enter to continue..."
    else:
        message = f"This script will attempt to repair your dataset {config.dataset_path}.\nPress enter to continue..."

    print(message)
    input()

    # indexes to delete
    if config.indexes_to_delete is not None:
        indexes_to_delete = config.indexes_to_delete.split(",")
        print(f"Indexes to delete: {indexes_to_delete}")

        delete_indexes([int(index) for index in indexes_to_delete])

    # Delete all the .DS_Store files in the dataset
    delete_DS_Store(dataset_path)

    # Process parquet files
    process_parquet_files(parquets_folder_path)

    # Run the stats script
    run_stats_script(dataset_path)

    # Push dataset to HF
    print("Pushing dataset to HF...")
    dataset_name = os.path.basename(dataset_path)
    subprocess.run(
        [
            "uv",
            "run",
            "push_dataset_to_hf.py",
            dataset_path,
            dataset_name,
        ],
        check=True,
    )
    print("Dataset pushed successfully")
