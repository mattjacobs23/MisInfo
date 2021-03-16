# MisInfo
A misinformation classifier powered by machine learning

![demo](MisInfo_loopingImage.gif)

This is a preliminary version of the project, trained on a smaller dataset. To see the bias present in this dataset, please see my exploratory data analysis [here](https://github.com/mattjacobs23/MisInfo/blob/main/notebooks/EDA_DataCleaning.ipynb))

## Features
* **Random forest classifier** powered by [Scikit-learn](https://scikit-learn.org/stable/).
* **RoBERTa** model powered by [HuggingFace Transformers](https://huggingface.co/transformers/) and [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning).
* **Data versioning** and configurable train/test pipelines using [DVC](https://github.com/iterative/dvc).
* **Exploratory data analysis** using [Pandas](https://pandas.pydata.org/).
* **Experiment tracking** and **logging** via [MLFlow](https://mlflow.org/).
* **Continuous integration** with [Github actions](https://github.com/features/actions).
* **Functionality tests** powered by [PyTest](https://docs.pytest.org/en/stable/) and [Great Expectations](https://greatexpectations.io/).
* **Error** and **model feature analysis** via [SHAP](https://github.com/slundberg/shap).
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
