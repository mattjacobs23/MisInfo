"""
Here we will be checking to see that the output of our functions have the appropriate shape,
that they are in the correct range (i.e. ≤1 if probabilities),
and that we can overfit a small train set
"""
import sys,os
sys.path.append(os.getcwd())


import numpy as np
import pytest

from MisInfo.models.tree_model import RandomForestModel
from MisInfo.preprocessing.feature_eng import Datum

# Decorate our config() and sample_data functions with a pytest fixture which prepares them for testing
@pytest.fixture
def config():
    return {
        "evaluate": False,
        "model_output_path": "",
        "featurizer_output_path": "",
        "credit_bins_path": "tests/fixtures/optimal_credit_bins.json",
        "params": {}
    }

@pytest.fixture
def sample_data():
    return [
        Datum(statement="sample statement 1 asd as",
                  barely_true_count=1,
                  false_count=1,
                  half_true_count=1,
                  mostly_true_count=1,
                  pants_fire_count=1,
                  subject="",
                  speaker="",
                  speaker_title="",
                  state_info="",
                  party_affiliation="",
                  label=True),
        Datum(statement="sample statement 2 asfa",
                  barely_true_count=2,
                  false_count=2,
                  half_true_count=2,
                  mostly_true_count=2,
                  pants_fire_count=2,
                  subject="",
                  speaker="",
                  speaker_title="",
                  state_info="",
                  party_affiliation="",
                  label=False),
        Datum(statement="sample statement 3 as dfa",
                  barely_true_count=3,
                  false_count=3,
                  half_true_count=3,
                  mostly_true_count=3,
                  pants_fire_count=3,
                  subject="",
                  speaker="",
                  speaker_title="",
                  state_info="",
                  party_affiliation="",
                  label=True)
    ]

# Have this function request our two fixtures above, train on the sample data and make sure we overfit
def test_rf_overfits_small_dataset(config, sample_data):
    model = RandomForestModel(config=config)
    train_labels = [True, False, True]

    model.train(sample_data)
    predicted_labels = np.argmax(model.predict(sample_data), axis=1)
    predicted_labels = list(map(lambda x: bool(x), predicted_labels))
    assert predicted_labels == train_labels

# Request the two fixtures, verify shape of predicted labels
def test_rf_correct_predict_shape(config, sample_data):
    model = RandomForestModel(config=config)

    model.train(sample_data)
    predicted_labels = np.argmax(model.predict(sample_data), axis=1)

    assert predicted_labels.shape[0] == 3

# Request the two fixtures, verify probabilities are <= 1
def test_rf_correct_predict_range(config, sample_data):
    model = RandomForestModel(config=config)

    model.train(sample_data)
    predicted_probs = model.predict(sample_data)

    assert (predicted_probs <= 1).all()
