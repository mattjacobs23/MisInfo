"""
Simple feature tests to verify our functions in feature_eng work as intended.
"""
import sys,os
sys.path.append(os.getcwd())

from MisInfo.preprocessing.feature_eng import credit_bins
from MisInfo.preprocessing.feature_eng import normalize_and_clean_counts
from MisInfo.preprocessing.feature_eng import normalize_and_clean_party_affiliations
from MisInfo.preprocessing.feature_eng import normalize_and_clean_speaker_title
from MisInfo.preprocessing.feature_eng import normalize_and_clean_state_info
from MisInfo.preprocessing.feature_eng import normalize_labels

# Credit scores were binned into 10 bins. Run simple test to make sure bins correctly.
def test_credit_bins():
    bins = [0, 4, 10, 12]
    assert credit_bins(0, bins) == 0
    assert credit_bins(3, bins) == 1
    assert credit_bins(4, bins) == 1
    assert credit_bins(12, bins) == 3

# Our normalize_labels function turns the labels into binary labels. Check with simple assertion here.
def test_normalize_labels():
    data = [
        {"label": "pants-fire", "ignored_field": "blah"},
        {"label": "barely-true"},
        {"label": "false"},
        {"label": "true"},
        {"label": "half-true"},
        {"label": "mostly-true"}
    ]

    expected_converted_data = [
        {"label": False, "ignored_field": "blah"},
        {"label": False},
        {"label": False},
        {"label": True},
        {"label": True},
        {"label": True}
    ]

    assert normalize_labels(data) == expected_converted_data

# Verify credit counts are converted to floats
def test_normalize_counts():
    data = [
        {"barely_true_count": "23.0", "ignored_label": "true"},
        {"false_count": "1.0"}
    ]

    expected_converted_data = [
        {"barely_true_count": 23.0, "ignored_label": "true"},
        {"false_count": 1.0}
    ]

    assert normalize_and_clean_counts(data) == expected_converted_data

# Verify state info entries are being normalized properly
def test_normalize_state_info():
    data = [
        {"state_info": " Virgina ", "ignored_label": "true"},
        {"state_info": " TEX "}
    ]

    expected_converted_data = [
        {"state_info": "virginia", "ignored_label": "true"},
        {"state_info": "texas"}
    ]

    assert normalize_and_clean_state_info(data) == expected_converted_data



# Verify speaker titles are being normalized properly
def test_normalize_speaker_title():
    data = [
        {"speaker_title": "mr-president ", "ignored_label": "true"},
        {"speaker_title": "  U. S. CONGRESSMAN"}
    ]

    expected_converted_data = [
        {"speaker_title": "mr president", "ignored_label": "true"},
        {"speaker_title": "u.s. congressman"}
    ]

    assert normalize_and_clean_speaker_title(data) == expected_converted_data

# Verify party affiliations are being normalized properly
def test_normalize_party_affiliations():
    data = [
        {"party_affiliation": "democrat", "ignored_label": "true"},
        {"party_affiliation": "boston tea"}
    ]

    expected_converted_data = [
        {"party_affiliation": "democrat", "ignored_label": "true"},
        {"party_affiliation": "none"}
    ]

    assert normalize_and_clean_party_affiliations(data) == expected_converted_data

