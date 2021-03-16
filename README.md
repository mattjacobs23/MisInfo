# MisInfo
A misinformation classifier powered by machine learning

![demo](gifs/MisInfo_loopingImage.gif)

This is a **preliminary** version of the project, trained on a small dataset (~10k datapoints). 

To gain a better undertsanding of this dataset, including what biases are present, please see my exploratory data analysis [here](https://github.com/mattjacobs23/MisInfo/blob/main/notebooks/EDA_DataCleaning.ipynb)

## Features
* **Exploratory data analysis** using [Pandas](https://pandas.pydata.org/).
* **Random forest classifier** powered by [Scikit-learn](https://scikit-learn.org/stable/).
* **RoBERTa** model powered by [HuggingFace Transformers](https://huggingface.co/transformers/) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
* **Data versioning** and configurable train/test pipelines using [DVC](https://github.com/iterative/dvc).
* **Experiment tracking** and **logging** via [MLFlow](https://mlflow.org/).
* **Error** and **model feature analysis** via [SHAP](https://github.com/slundberg/shap).
* **Functionality tests** powered by [PyTest](https://docs.pytest.org/en/stable/) and [Great Expectations](https://greatexpectations.io/).
* **Continuous integration** with [Github actions](https://github.com/features/actions).
* **Production-ready server** via [FastAPI](https://fastapi.tiangolo.com/) and [Gunicorn](https://gunicorn.org/).
* **Chrome extension** for interacting with a model in the [browser](https://chrome.google.com/webstore/category/extensions?hl=en).

## How to Use It
Go to the root directory of the repo and start a virtual environment. Then run:
```
pip install -r requirements.txt
```

You can download the LIAR dataset using [this link](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip), which includes 12.8k human labeled short statements from POLITIFACT.COM's API. Each statement is evaluated by a POLITIFACT.COM editor for its truthfulness.

Move the test.tsv, train.tsv, and valid.tsv to data/raw

You're ready to go!

### Train
To train the [random forest baseline](https://github.com/mattjacobs23/MisInfo/blob/main/MisInfo/models/tree_model.py), run the following from the root directory:
```
dvc repro train-random-forest
```

Your output should look something like the following:
```
INFO - 2021-03-15 18:04:40,389 - feature_eng.py - Creating featurizer from scratch...
INFO - 2021-03-15 18:04:40,394 - tree_model.py - Initializing model from scratch...
INFO - 2021-03-15 18:04:40,394 - train.py - Training model...
INFO - 2021-03-15 18:04:40,722 - feature_eng.py - Saving featurizer to disk...
INFO - 2021-03-15 18:04:40,727 - tree_model.py - Featurizing data from scratch...
INFO - 2021-03-15 18:04:49,568 - tree_model.py - Saving model to disk...
INFO - 2021-03-15 18:04:49,635 - train.py - Evaluating model...
INFO - 2021-03-15 18:04:49,776 - train.py - Val metrics: {'val f1': 0.765313145216793, 'val accuracy': 0.7344236760124611, 'val auc': 0.8168537114083521, 'val true negative': 387, 'val false negative': 112, 'val false positive': 229, 'val true positive': 556}
INFO - 2021-03-15 18:04:49,905 - train.py - Test metrics: {'test f1': 0.7817460317460316, 'test accuracy': 0.739542225730071, 'test auc': 0.8083942437734588, 'test true negative': 346, 'test false negative': 123, 'test false positive': 207, 'test true positive': 591}
```

### Deploy

Once you have successfully trained a model using the step above, you should have a model checkpoint saved in `model_checkpoints/random_forest`.

Now build your deployment Docker image:
```
docker build . -f deploy/Dockerfile.serve -t misinfo-deploy
```

Once your image is built, you can run the model locally via a REST API with:
```
docker run -p 8000:80 -e MODEL_DIR="/home/MisInfo/random_forest" -e MODULE_NAME="MisInfo.server_logic.main" misinfo-deploy
```

From here you can interact with the API using [Postman](https://www.postman.com/) or through a simple cURL request:
```
curl -X POST http://127.0.0.1:8000/api/predict-misinfo -d '{"text": "Some example string."}'
```

### Chrome Extension

To interact with the model in the browser, go to Extensions > Manage Extensions. You can also click on [this link](chrome://extensions/).

Then click on "Load Unpacked" near the top left, and select the **deploy/extension** folder from the repository. 

Once this extension is enabled you should be able to highlight text in your Chrome browser and recieve classification predictions.

### Credits

Special thanks to [Mihail Eric](https://www.mihaileric.com/) for drastically improving my understanding of machine learning architecture. 
