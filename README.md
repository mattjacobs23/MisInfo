# MisInfo
A misinformation classifier powered by machine learning

## Motivation
The recent proliferation of social media and other digital spaces has significantly changed the way in which people acquire information. According to the 2020 Pew Research Center survey, about half of U.S. adults (53%) say they get news from social media at least occasionally. Disingenuous information can spread virally and often faster than true information. In many of cases, even when correct information later disseminates, the rapid spread of misinformation can have devastating consequences. Therefore, there is an urgent need for the development of automatic detection of misinformation to help stop the viral spread of such news. 

MisInfo aims to be a misinformation detection system which can detect truthfulness of snippets of text from different sources including political debates, social media platforms, etc. It can then provide an indication of which headlines should invoke an extra level of caution:

![Demo](gifs/cropLoopingImage2.gif)

This is a **preliminary** version of the project, trained on a small dataset (~12k datapoints). 

To gain a better undertsanding of this dataset, including what biases are present, please see my exploratory data analysis [here](https://github.com/mattjacobs23/MisInfo/blob/main/notebooks/EDA_DataCleaning.ipynb)

I am currently trying to use the NELA-GT-2019 dataset, a large multi-labelled news dataset, found [here](https://github.com/mgruppi/nela-gt-2019). Deep learning (RoBERTa) did not provide better results than the base random forest classifier on the small LIAR dataset, so I am currently trying deep learning techniques on the larger dataset.

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

You can also enter extra parameters for the model in the configuration file, found in config/random_forest.json.

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

To interact with the model in the browser, go to Extensions > Manage Extensions. You can also type chrome://extensions/ into Chrome's Omnibox and press Enter.

Then click on "Load Unpacked" near the top left, and select the **deploy/extension** folder from the repository. 

Once this extension is enabled you should be able to highlight text in your Chrome browser and recieve classification predictions.

## Credits

Special thanks to [Mihail Eric](https://www.mihaileric.com/) for drastically improving my understanding of machine learning architecture. 

## About Me
I have a master of science in physics and am transitioning to the AI/Machine Learning industry. If you have any questions about me, this project, or anything else, feel free to reach out: 
* [LinkedIn](https://www.linkedin.com/in/matt-jacobs-23007/)
