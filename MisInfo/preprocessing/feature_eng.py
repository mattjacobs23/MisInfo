'''
Here we extract manual features and ngram features.
We then concatenate them into a single vector to capture all the most salient information we want to capture from our data and provide to our model.
We normalize and clean the data, which should be passed in as List[Dict]

'''
import sys,os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
sys.path.append(os.getcwd())

import json
import logging
import os
import pickle
from copy import deepcopy
from functools import partial
from typing import Dict
from typing import List
from typing import Optional

import math
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Our predefined canonical transformation dictionaries
from MisInfo.preprocessing.canonical import CANONICAL_SPEAKER_TITLES
from MisInfo.preprocessing.canonical import CANONICAL_STATE
from MisInfo.preprocessing.canonical import PARTY_AFFILIATIONS
from MisInfo.preprocessing.canonical import SIX_WAY_LABEL_TO_BINARY

'''
    Have all Python modules can participate in logging, so application log can include
        my own messages integrated with messages from third-party modules.

    Logging allows writing status messages to a file or any other output streams.
    The file can contain the information on which part of the code is executed and what problems have been arisen
'''
logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

# Good engineering practice is to strictly define the types of each datum constituent
# Untrusted data can be passed to a model, and after parsing and validation pydantic guarantees that the fields of the resultant model instance will conform to the field types defined on the model
class Datum(BaseModel):
    statement_json: Optional[str]
    label: Optional[bool]
    statement: str
    subject: Optional[str]
    speaker: Optional[str]
    speaker_title: Optional[str]
    state_info: Optional[str]
    party_affiliation: Optional[str]
    barely_true_count: float
    false_count: float
    half_true_count: float
    mostly_true_count: float
    pants_fire_count: float
    context: Optional[str]
    justification: Optional[str]


# In EDA we discovered that the distriutions of "credit history" could be useful information for our model
# The counts can be quite spread out, so we will bin the values into 10 bins as follows
def credit_bins(val: float, bins: List[float]) -> int:
    for index, bin_val in enumerate(bins):
        if val <= bin_val:
            return index

# One-hot-encode speaker, speaker_title, state_info, party_affiliation. Also extract binned credit scores.
def manual_features(data: List[Datum], optimal_credit_bins: Dict) -> List[Dict]:
    all_features = []
    for datum in data:
        features = {}
        # Add respective datum constituents to the feature dictionary
        features["speaker"] = datum.speaker
        features["speaker_title"] = datum.speaker_title
        features["state_info"] = datum.state_info
        features["party_affiliation"] = datum.party_affiliation
        # Compute credit score features by looping through credit names,
        datum = dict(datum)
        for feat in ["barely_true_count", "false_count", "half_true_count", "mostly_true_count", "pants_fire_count"]:
            features[feat] = str(credit_bins(datum[feat], optimal_credit_bins[feat]))
        all_features.append(features) # all_features will be a list of dictionaries
    return all_features


'''
Ngram features are often useful when dealing with text as they allow us to pick up on certain lexical and linguistic patterns in the data.

We will use tfidf-weights for these features rather than raw ngrams as these types of weight are often using in information retrieval to help upweight/downweight the importance of certain words.
'''

def extract_statements(data: List[Datum]) -> List[str]:
    return [datum.statement for datum in data]

def construct_datum(input: str) -> Datum:
    return Datum(**{
        "statement": input,
        "barely_true_count": float("nan"),
        "false_count": float("nan"),
        "half_true_count": float("nan"),
        "mostly_true_count": float("nan"),
        "pants_fire_count": float("nan"),
    })

'''
Define a separate Featurizer class which will expose an interface for training the featurizer (necessary for the ngram-based weights) and featurizing arbitrary data:
The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.
A lot of the work done with ngram features is handled for us by libraries (like Scikit-learn).
'''
# Using inheretance from the class object (inhereting all methods and properties)
class Featurizer(object):
    def __init__(self, featurizer_cache_path: str, config: Optional[Dict] = None):
        # NOTE: Here you can add feature caching which helps if it's too expensive to compute features from scratch for each run
        # Can pass configuration, where we can load optimal credit bins from. Optional[Dict] telling the type checker that either an object of the specific type is required, or None is required
        # First check if featurizer cache path already exists. If so, open that cached feature file. Else we compute features from scratch and write to that path.
        if os.path.exists(featurizer_cache_path):
            LOGGER.info("Loading featurizer from cache...")
            # Use "rb" method to open the file in binary format for reading
            with open(featurizer_cache_path, "rb") as f:
                self.combined_featurizer = pickle.load(f)
        else:
            LOGGER.info("Creating featurizer from scratch...")
            # Obtain directory name from the absolutized file path (the absolute path contains the root directory and all other subdirectories where a file or folder is contained)
            # Need parent directory in a source tree that is multiple directories up, call dirname multiple times
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            # Load optimal credit bins
            with open(os.path.join(base_dir, config["credit_bins_path"])) as f:
                optimal_credit_bins = json.load(f)
            # Use scikit learn feature_extraction methods
            # DictVectorizer transforms lists of feature-value mappings (dict-like objects) to vectors
            dict_featurizer = DictVectorizer()
            # TfidfVectorizer we used in EDA, converts a collection of raw documents to a matrix of TF-IDF features
            tfidf_featurizer = TfidfVectorizer()

            # A FunctionTransformer forwards its X (and optionally y) arguments to a user-defined function or function object and returns the result of this function
            statement_transformer = FunctionTransformer(extract_statements)
            manual_feature_transformer = FunctionTransformer(partial(manual_features,
                                                                     optimal_credit_bins=optimal_credit_bins))

            # Sklearn's Pipeline() allows sequential application of a list of transforms and a final estimator
            manual_feature_pipeline = Pipeline([
                ("manual_features", manual_feature_transformer), # manual_feature_transformer extracts the manual features
                ("manual_featurizer", dict_featurizer) # dict_featurizer is a DictVectorizer() object
            ])

            ngram_feature_pipeline = Pipeline([
                ("statements", statement_transformer), # statement_transformer extracts the statements
                ("ngram_featurizer", tfidf_featurizer) # tfidf_featurizer is a TfidfVectorizer() object
            ])

            # FeatureUnion concatenates results of multiple transformer objects into a single transformer
            # This estimator applies a list of transformer objects in parallel to the input data, then concatenates the results
            self.combined_featurizer = FeatureUnion([
                ("manual_feature_pipe", manual_feature_pipeline), # tuple of name and the actual pipeline
                ("ngram_feature_pipe", ngram_feature_pipeline)
            ])

    # Grab all of the feature names
    def get_all_feature_names(self) -> List[str]:
        all_feature_names = []
        # Loop through the transformer tuples in combined_featurizer
        for name, pipeline in self.combined_featurizer.transformer_list:
            # Grab the names inside the manual and ngram pipelines
            final_pipe_name, final_pipe_transformer = pipeline.steps[-1]
            # Add each feature name to the end of the list
            all_feature_names.extend(final_pipe_transformer.get_feature_names())
        return all_feature_names

    def fit(self, data: List[Datum]) -> None:
        # Fit all transformers using the data
        self.combined_featurizer.fit(data)

    def featurize(self, data: List[Datum]) -> np.array:
        # Transform the data and concatenate results
        return self.combined_featurizer.transform(data)

    def save(self, featurizer_cache_path: str):
        LOGGER.info("Saving featurizer to disk...")
        with open(featurizer_cache_path, "wb") as f:
            pickle.dump(self.combined_featurizer, f)


'''
Now normalize and clean each of the features one at a time
Making sure that all normalization operations preserve immutability of inputs
A deep copy constructs a new compound object and then, recursively, inserts copies into it of the objects found in the original.
Deepcopy() keeps a memo dictionary of objects already copied during the current copying pass;
    and lets user-defined classes override the copying operation or the set of components copied
'''
# 1)
# Turn labels into binary labels, make everything lowercase and strip of extra whitespace
def normalize_labels(data: List[Dict]) -> List[Dict]:
    normalized_data = []
    for datum in data:
        # First do simple cleaning
        normalized_datum = deepcopy(datum) # preserve immutability of input data
        # Use our predefined dictionary
        normalized_datum["label"] = SIX_WAY_LABEL_TO_BINARY[datum["label".lower().strip()]]
        normalized_data.append(normalized_datum)
    return normalized_data

# 2)
# Grab the credit counts as floats
def normalize_and_clean_counts(data: List[Dict]) -> List[Dict]:
    normalized_data = []
    for index, datum in enumerate(data):
        normalized_datum = deepcopy(datum) # preserve immutability of input data
        for count_col in ["barely_true_count",
                          "false_count",
                          "half_true_count",
                          "mostly_true_count",
                          "pants_fire_count"]:
            # First check if that this Datum has that particular column. Can allow use of future data which does not have these columns.
            if count_col in normalized_datum:
                # Cannot pass Nonetype values to float(). If NaN entry we set this to 0
                if normalized_datum[count_col] == None:
                    normalized_datum[count_col] = float(0)
                # Otherwise set the string entry to be floating type
                normalized_datum[count_col] = float(normalized_datum[count_col])
        # Add this normalized datum (Dict) to the normalized data array
        normalized_data.append(normalized_datum)
    return normalized_data

# 3)
# Convert state info to canonical form, strip of whitespace, lowercase the field, and replace dashes with spaces
def normalize_and_clean_state_info(data: List[Dict]) -> List[Dict]:
    normalized_data = []
    for datum in data:
        normalized_datum = deepcopy(datum) # preserve immutability of input data
        old_state_info = normalized_datum["state_info"]
        # We have some NaN values (Nonetype) in dataset, cannot pass that to .lower() method etc. Give them value "none"
        if old_state_info == None:
            old_state_info = "Unknown"
        new_state_info = old_state_info.lower().strip().replace("-", " ")
        # Check to see if this cleaned state_info datum is in our predefined canonical dictionary
        if new_state_info in CANONICAL_STATE:
            # Set it to its canonical form
            new_state_info = CANONICAL_STATE[new_state_info]
        # Enter the cleaned state_info into our new normalized datum
        normalized_datum["state_info"] = new_state_info
        # Add this cleaned datum to the new normalized dataset
        normalized_data.append(normalized_datum)
    return normalized_data

# 4)
# Convert speaker title to canonical form, lowercase the field, strip whitespace, replace dashes
def normalize_and_clean_speaker_title(data: List[Dict]) -> List[Dict]:
    normalized_data = []
    for datum in data:
        # First do simple cleaning
        normalized_datum = deepcopy(datum) # preserve immutability of input data
        old_speaker_title = normalized_datum["speaker_title"]
        # We have some NaN values (Nonetype) in dataset, cannot pass that to .lower() method etc. Give them value "none"
        if old_speaker_title == None:
            old_speaker_title = "Unknown"
        new_speaker_title = old_speaker_title.lower().strip().replace("-", " ")
        # Then canonicalize
        if new_speaker_title in CANONICAL_SPEAKER_TITLES:
            new_speaker_title = CANONICAL_SPEAKER_TITLES[new_speaker_title]
        normalized_datum["speaker_title"] = new_speaker_title
        normalized_data.append(normalized_datum)
    return normalized_data

# 5)
# Convert party affiliation to canonical form
def normalize_and_clean_party_affiliations(data: List[Dict]) -> List[Dict]:
    normalized_data = []
    for datum in data:
        normalized_datum = deepcopy(datum)
        # Check if datum in canonical dictionary, if not, give it value "none"
        if normalized_datum["party_affiliation"] not in PARTY_AFFILIATIONS:
            normalized_datum["party_affiliation"] = "none"
        normalized_data.append(normalized_datum)
    return normalized_data

# 6)
# Almost all of the justification column is NaN
def remove_justification_col(data: List[Dict]) -> List[Dict]:
    normalized_data = []
    for datum in data:
        normalized_datum = deepcopy(datum) # preserve immutability of input data
        if 'justification' in normalized_datum:
            del normalized_datum['justification']
        normalized_data.append(normalized_datum)
    return normalized_data

# 7)
# There are a few NaN values in the context column. Future work could involve binning the context entries, as right now there are over 4000 unique values in that column.
def normalize_and_clean_context(data: List[Dict]) -> List[Dict]:
    normalized_data = []
    for datum in data:
        normalized_datum = deepcopy(datum)
        old_context = normalized_datum['context']
        if old_context == None:
            old_context = "Unknown"
        new_context = old_context.lower().strip().replace("-", " ")
        normalized_datum["context"] = new_context
        normalized_data.append(normalized_datum)
    return normalized_data

# 8)
# There are a few NaN values in the subject column.
def normalize_and_clean_subject(data: List[Dict]) -> List[Dict]:
    normalized_data = []
    for datum in data:
        normalized_datum = deepcopy(datum)
        old_subject = normalized_datum['subject']
        if old_subject == None:
            old_subject = "Unknown"
        new_subject = old_subject.lower().strip().replace("-", " ")
        normalized_datum["subject"] = new_subject
        normalized_data.append(normalized_datum)
    return normalized_data

# 9)
# There are a few NaN values in the speaker column.
def normalize_and_clean_speaker(data: List[Dict]) -> List[Dict]:
    normalized_data = []
    for datum in data:
        normalized_datum = deepcopy(datum)
        old_speaker = normalized_datum['speaker']
        if old_speaker == None:
            old_speaker = "Unknown"
        new_speaker = old_speaker.lower().strip().replace("-", " ")
        normalized_datum["speaker"] = new_speaker
        normalized_data.append(normalized_datum)
    return normalized_data

# Run each of the normalization and cleaning functions on the data. Total of 9 functions
def normalize_and_clean(data: List[Dict]) -> List[Dict]:
    return normalize_and_clean_speaker(
        normalize_and_clean_subject(
            normalize_and_clean_context(
                remove_justification_col(
                    normalize_and_clean_party_affiliations(
                        normalize_and_clean_speaker_title(
                            normalize_and_clean_state_info(
                                normalize_and_clean_counts(
                                    normalize_labels(
                                        data
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
