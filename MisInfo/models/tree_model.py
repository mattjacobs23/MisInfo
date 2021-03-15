'''
Random Forest base model
  - A random forest is a meta estimator that fits a number of decision tree classifiers on various
    sub-samples of the dataset and uses averaging to improve the predictive accuracy and control
    over-fitting.
'''
import sys,os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.getcwd())

import logging
import os
import pickle
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# Sometimes has difficulty recognizing "MisInfo" module in front of these
from MisInfo.preprocessing.feature_eng import Datum
from MisInfo.preprocessing.feature_eng import Featurizer
from MisInfo.models.base import Model

# set up the same logging configuration
logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

"""
Can initialize model with a *configuration file*, which is a json file including what model to use,
the train/val/test data paths, featurizer output path, credit bins path, model output path
evaluate (boolean), and any extra paramaters the user wants to include
"""

class RandomForestModel(Model):
    def __init__(self, config: Optional[Dict] = None):
        # Allow user to specify a configuration if desired
        self.config = config
        # Initiate a path to cache model paraemeters
        model_cache_path = os.path.join(config["model_output_path"], "model.pkl")
        # Initiate featurizer for the model
        self.featurizer = Featurizer(os.path.join(config["featurizer_output_path"],
                                                      "featurizer.pkl"),
                                         config)
        # Include validation of model output path
        if "evaluate" in config and config["evaluate"] and not os.path.exists(model_cache_path):
            raise ValueError("Model output path does not exist but in 'evaluate' mode!")
        # Include validation of model cache path
        if model_cache_path and os.path.exists(model_cache_path):
            LOGGER.info("Loading model from cache...")
            with open(model_cache_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            LOGGER.info("Initializing model from scratch...")
            self.model = RandomForestClassifier(**self.config["params"])

    # Train method for random forest classifier takes in train/val data,
    # and cache_featurizer (boolean, whether or not to cache the model featurizer)
    def train(self,
              train_data: List[Datum],
              val_data: List[Datum] = None,
              cache_featurizer: Optional[bool] = False) -> None:
        # Fit the initialized featurizer on the train data
        self.featurizer.fit(train_data)
        # If user wishes to cache model featurizer, save the feature_names
        if cache_featurizer:
            # Use our get_all_feature_names method of the class featurizer
            feature_names = self.featurizer.get_all_feature_names()
            # Open the file in binary format for writing
            with open(os.path.join(self.config["model_output_path"],
                                   "feature_names.pkl"), "wb") as f:
                pickle.dump(feature_names, f)
            # Use our save method which also performs pickle.dump()
            self.featurizer.save(os.path.join(self.config["featurizer_output_path"],
                                              "featurizer.pkl"))
        # Extract labels from train data
        train_labels = [datum.label for datum in train_data]
        LOGGER.info("Featurizing data from scratch...")
        # The featurize method transforms the data and concatenates the results
        train_features = self.featurizer.featurize(train_data)
        # Fit all transformers on the train data
        self.model.fit(train_features, train_labels)

    # Method to compute f1, accuracy, auc, and confusion matrix
    def compute_metrics(self, eval_data: List[Datum], split: Optional[str] = None) -> Dict:
        # Extract the evaluation labels
        expected_labels = [datum.label for datum in eval_data]
        # Perform predictions on evaluation data
        predicted_proba = self.predict(eval_data)
        # Extract predicted label as largest predicted probability value
        predicted_labels = np.argmax(predicted_proba, axis=1)

        # Use sklearn metrics functions
        accuracy = accuracy_score(expected_labels, predicted_labels)
        f1 = f1_score(expected_labels, predicted_labels)
        auc = roc_auc_score(expected_labels, predicted_proba[:, 1])
        conf_mat = confusion_matrix(expected_labels, predicted_labels)
        tn, fp, fn, tp = conf_mat.ravel()

        split_prefix = "" if split is None else split
        return {
            f"{split_prefix} f1": f1,
            f"{split_prefix} accuracy": accuracy,
            f"{split_prefix} auc": auc,
            f"{split_prefix} true negative": tn,
            f"{split_prefix} false negative": fn,
            f"{split_prefix} false positive": fp,
            f"{split_prefix} true positive": tp,
        }

    def predict(self, data: List[Datum]) -> np.array:
        # Extract features from featurize method, which transforms the data and concatenates the results
        features = self.featurizer.featurize(data)
        return self.model.predict_proba(features)

    def get_params(self) -> Dict:
        return self.model.get_params()

    def save(self, model_cache_path: str) -> None:
        LOGGER.info("Saving model to disk...")
        # Could also try saving using joblib in future
        with open(model_cache_path, "wb") as f:
            pickle.dump(self.model, f)

