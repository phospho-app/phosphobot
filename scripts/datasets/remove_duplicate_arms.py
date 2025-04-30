import os
import json
import argparse
import subprocess
import pandas as pd
from pathlib import Path


def edit_parquets_to_remove_duplicate_arms(folder_path):
    print(f"Editing parquet files in {folder_path} to remove duplicate arms...")
    edited_data_length = None
    for file in Path(folder_path).glob("*.parquet"):
        # Read parquet file
        df = pd.read_parquet(file)

        # Slice the last 6 items from the arrays in "action" and "observation.state"
        df["action"] = df["action"].apply(lambda x: x[: len(x) // 2])
        df["observation.state"] = df["observation.state"].apply(
            lambda x: x[: len(x) // 2]
        )
        if edited_data_length is None:
            edited_data_length = len(df["action"][0])

        # Overwrite the original file with the modified data
        df.to_parquet(file)

        print(f"Processed {file.name}")

    if edited_data_length is None:
        raise ValueError("No parquet files found in the specified folder.")

    print("All parquets processed.")
    # Return the new number of arms in the dataset, should be 1 or 2 in 99% of cases
    return edited_data_length // 6


def edit_info_json_path(info_file_path, number_of_arms):
    print(f"Editing info.json file at {info_file_path} to remove duplicate arms...")
    with open(info_file_path, "r") as f:
        data = json.load(f)
    if number_of_arms == 1:
        robot_type = "so-100"
        action_shape = [6]
        action_names = [
            "motor_1",
            "motor_2",
            "motor_3",
            "motor_4",
            "motor_5",
            "motor_6",
        ]
        observation_shape = [6]
        observation_names = [
            "motor_1",
            "motor_2",
            "motor_3",
            "motor_4",
            "motor_5",
            "motor_6",
        ]
    elif number_of_arms == 2:
        robot_type = "so-100, so-100"
        action_shape = [12]
        action_names = [
            "motor_1",
            "motor_2",
            "motor_3",
            "motor_4",
            "motor_5",
            "motor_6",
            "motor_1_secondary",
            "motor_2_secondary",
            "motor_3_secondary",
            "motor_4_secondary",
            "motor_5_secondary",
            "motor_6_secondary",
        ]
        observation_shape = [12]
        observation_names = [
            "motor_1",
            "motor_2",
            "motor_3",
            "motor_4",
            "motor_5",
            "motor_6",
            "motor_1_secondary",
            "motor_2_secondary",
            "motor_3_secondary",
            "motor_4_secondary",
            "motor_5_secondary",
            "motor_6_secondary",
        ]
    else:
        raise ValueError(
            "This dataset doesn't seem to be compatible with the current version of the script. Please check the number of arms in the dataset."
        )
    data["robot_type"] = robot_type
    data["features"]["action"]["shape"] = action_shape
    data["features"]["action"]["names"] = action_names
    data["features"]["observation.state"]["shape"] = observation_shape
    data["features"]["observation.state"]["names"] = observation_names

    # Overwrite the original file with the modified data
    with open(info_file_path, "w") as f:
        json.dump(data, f, indent=4)

    print("info.json processed.")


def compute_stats(dataset_path):
    print(f"Computing statistics for dataset at {dataset_path}...")
    subprocess.run(
        [
            "uv",
            "run",
            "lerobot_stats_compute.py",
            "--dataset-path",
            DATASET_PATH,
        ],
        check=True,
    )
    print("Statistics computed.")


def push_to_hub():
    print("Pushing your new dataset to Hugging Face Hub...")
    dataset_name = os.path.basename(DATASET_PATH) + "_edited"
    subprocess.run(
        [
            "uv",
            "run",
            "push_dataset_to_hf.py",
            DATASET_PATH,
            dataset_name,
        ],
        check=True,
    )
    print("Pushed to Hugging Face Hub.")
    return dataset_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for a dataset.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory containing data, videos, and meta subfolders.",
    )
    args = parser.parse_args()

    DATASET_PATH = args.dataset_path

    PARQUETS_PATH = os.path.join(DATASET_PATH, "data", "chunk-000")
    new_number_of_arms = edit_parquets_to_remove_duplicate_arms(PARQUETS_PATH)

    INFO_FILE_PATH = os.path.join(DATASET_PATH, "meta", "info.json")
    edit_info_json_path(INFO_FILE_PATH, new_number_of_arms)

    compute_stats(DATASET_PATH)

    dataset_name = push_to_hub()

    print("Your edited dataset is ready and available on Hugging Face Hub")
