"""
Great Expectations also supports running data suite directly from the commandline,
but we are using this Python script we can invoke manually,
better for continuous integration system.
"""

import os
from datetime import datetime

import great_expectations as ge

# A DataContext represents a Great Expectations project.
# It organizes storage and access for expectation suites, datasources, notification settings, and data fixtures.
context = ge.data_context.DataContext()

datasource_name = "misinfo_data"

# In data_preprocessing.py we saved clean versions of the data. Will use those.
# Using our misinfo_data_suite.json file which is in ge format to verify every column of data
train_batch = context.get_batch(
    {"path": f"{os.environ['GE_DIR']}/data/processed/cleaned_train_data.json",
     "datasource": datasource_name},
    "misinfo_data_suite")
val_batch = context.get_batch(
    {"path": f"{os.environ['GE_DIR']}/data/processed/cleaned_val_data.json",
     "datasource": datasource_name},
    "misinfo_data_suite")
test_batch = context.get_batch(
    {"path": f"{os.environ['GE_DIR']}/data/processed/cleaned_test_data.json",
     "datasource": datasource_name},
    "misinfo_data_suite")

results = context.run_validation_operator(
    "action_list_operator",
    assets_to_validate=[train_batch, val_batch, test_batch],
    run_id=str(datetime.now()))

print(results)
if results["success"]:
    print("Test suite passed!")
    exit(0)
else:
    print("Test suite failed!")
    exit(1)
