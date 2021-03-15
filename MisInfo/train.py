import argparse
import json
import logging
import os
import sys
import random
from shutil import copy

import mlflow
import numpy as np
import torch

from models.tree_model import RandomForestModel
from preprocessing.reader import read_json_data

"""
Using MLflow: The mlflow module provides a high-level “fluent” API for starting and managing MLflow runs.
Super handy way to monitor training parameters, metrics, etc
MLflow Projects can organize and describe your code to let other data scientists (or automated tools) run it.
"""


logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

# Use argparse library to read in the arguments given by the user
def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Add the ability for the user to include their configuration file, all arguments should be included in that file
    parser.add_argument("--config-file", type=str)
    return parser.parse_args()

def set_random_seed(val: int = 1) -> None:
    random.seed(val)
    np.random.seed(val)
    # Torch-specific random-seeds
    torch.manual_seed(val)
    torch.cuda.manual_seed_all(val)

if __name__ == "__main__":
    # Use read_args() function defined above to read in arguments given in user's call
    args = read_args()
    # Open the configuration file specified by the user
    with open(args.config_file) as f:
        config = json.load(f)

    # Set a random seed using our function and constant input for consistent, reproducible results
    set_random_seed(42)
    # Set given experiment as active experiment. If experiment does not exist, create an experiment with provided name
    mlflow.set_experiment(config["model"])

    # Grab the base directory which will be the top MisInfo folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Set the full model output path using the configuration file.
    model_output_path = os.path.join(base_dir, config["model_output_path"])
    # Update full model output path
    config["model_output_path"] = model_output_path
    # Create a new directory, if exists already we leave the directory unaltered.
    os.makedirs(model_output_path, exist_ok=True)
    # Copy config to model directory
    copy(args.config_file, model_output_path)

    # Start a MLflow run
    with mlflow.start_run() as run:
        # Set output path to write mlflow file to
        with open(os.path.join(model_output_path, "meta.json"), "w") as f:
            json.dump({"mlflow_run_id": run.info.run_id}, f)
        # Log a batch of tags for the current run. We only need "evaluate" tag, has boolean value
        mlflow.set_tags({
            "evaluate": config["evaluate"]
        })

        # The base_dir is the top MisInfo folder
        train_data_path = os.path.join(base_dir, config["train_data_path"])
        val_data_path = os.path.join(base_dir, config["val_data_path"])
        test_data_path = os.path.join(base_dir, config["test_data_path"])
        # Read data
        train_data = read_json_data(train_data_path)
        val_data = read_json_data(val_data_path)
        test_data = read_json_data(test_data_path)

        # Check which model the user wishes to use (as detailed in config file)
        if config["model"] == "random_forest":
            config["featurizer_output_path"] = os.path.join(base_dir, config["featurizer_output_path"])
            model = RandomForestModel(config)
        elif config["model"] == "roberta":
            model = RobertaModel(config)
        else:
            raise ValueError(f"Invalid model type {config['model']} provided")

        # If evaluate set to False, we train the model on training data
        if not config["evaluate"]:
            LOGGER.info("Training model...")
            # Use train method of the specified model, featurizes and fits
            model.train(train_data, val_data, cache_featurizer=True)
            if config["model"] == "random_forest":
                # Cache model weights on disk
                model.save(os.path.join(model_output_path, "model.pkl"))

        # Log a batch of params for the current run.
        mlflow.log_params(model.get_params())
        LOGGER.info("Evaluating model...")
        # Use our compute_metrics method to get validation accuracy, f1, auc, confusion matrix
        val_metrics = model.compute_metrics(val_data, split="val")
        # Show metrics to user
        LOGGER.info(f"Val metrics: {val_metrics}")
        # Now for test data
        test_metrics = model.compute_metrics(test_data, split="test")
        LOGGER.info(f"Test metrics: {test_metrics}")
        # Log validation and test metrics for the current run
        mlflow.log_metrics(val_metrics)
        mlflow.log_metrics(test_metrics)
