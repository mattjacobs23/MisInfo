import logging
from typing import List

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseSettings
from pydantic.main import BaseModel

from MisInfo.models.tree_model import RandomForestModel
from MisInfo.preprocessing.feature_eng import construct_datum

# Set up logging as usual
logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

# Inherit from pydantic.BaseSettings class, which allows pydantic to be used in both a "validate this request data" context and in a "load my system settings" context.
# The model initialiser will attempt to determine the values of any fields not passed as keyword arguments by reading from the environment.
class Settings(BaseSettings):
    model_dir: str
# User will need to declare MODEL_DIR

# Initialization of FastAPI app
app = FastAPI()
# Initialization of settings object
settings = Settings()

# Enable CORS
# A "middleware" is a function that works with every request before it is processed by any specific path operation. And also with every response before returning it.
# CORS allows configuration of web API's security, allowing our web app (on different server) to make calls to our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
# Initialization of model calls for config which gives model output path and featurizer output path.
# The Featurizer class needs to call config["credit_bins_path"] however.
config = {
    "model_output_path": settings.model_dir,
    "featurizer_output_path": settings.model_dir,
    "credit_bins_path": "data/processed/optimal_credit_bins.json",
    "params": {}
}
model = RandomForestModel(config)

# Inherit from pydantic.main.BaseModel to perform parsing and validation of model instances
# The request body will be the statement text the user wishes to asses as misinformation or not.
class Statement(BaseModel):
    text: str

# The response body will be of this class Prediction. Allow BaseModel to perform parsing and validation.
class Prediction(BaseModel):
    label: float
    probs: List[float]

# Define a single REST endpoint called "/api/predict-misinfo"
# This injests a textual statement, runs inference on appropriately-defined datum, and outputs a prediction response object.
@app.post("/api/predict-misinfo", response_model=Prediction)
def predict_misinfo(statement: Statement):
    datum = construct_datum(statement.text)
    probs = model.predict([datum])
    label = np.argmax(probs, axis=1)
    prediction = Prediction(label=label[0], probs=list(probs[0]))
    LOGGER.info(prediction)
    return prediction
