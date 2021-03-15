# preprocessing data for classification of misinformation
import argparse
import csv
import json
import os
from typing import Dict
from typing import List

from feature_eng import normalize_and_clean

# use argparse to allow user to specify directories
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str)
    parser.add_argument("--val-data-path", type=str)
    parser.add_argument("--test-data-path", type=str)
    parser.add_argument("--output-dir", type=str)
    return parser.parse_args()

# define general function to read in data from a file and convert to a list of dictionaries we will use for normalization and cleaning
def read_datapoints(datapath: str) -> List[Dict]:
    with open(datapath) as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=[
            "statement_json",
            "label",
            "statement",
            "subject",
            "speaker",
            "speaker_title",
            "state_info",
            "party_affiliation",
            "barely_true_count",
            "false_count",
            "half_true_count",
            "mostly_true_count",
            "pants_fire_count",
            "context",
            "justification"
        ])
        return [row for row in reader]

if __name__ == "__main__":
    # read in the arguments given by the user
    args = read_args()

    # read in the train, validation, and test datasets
    train_data = read_datapoints(args.train_data_path)
    val_data = read_datapoints(args.val_data_path)
    test_data = read_datapoints(args.test_data_path)

    # apply the predefined normalization and cleaning function
    train_data = normalize_and_clean(train_data)
    val_data = normalize_and_clean(val_data)
    test_data = normalize_and_clean(test_data)

    # write the preprocessed data to json files
    with open(os.path.join(args.output_dir, "cleaned_train_data.json"), "w") as f:
        json.dump(train_data, f)

    with open(os.path.join(args.output_dir, "cleaned_val_data.json"), "w") as f:
        json.dump(val_data, f)

    with open(os.path.join(args.output_dir, "cleaned_test_data.json"), "w") as f:
        json.dump(test_data, f)
