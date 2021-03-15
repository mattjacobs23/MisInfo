'''
This is the base model class definition that all other models should inherit from
Just good software engineering practice to make model exploration easier.
'''

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from MisInfo.preprocessing.feature_eng import Datum


class Model(ABC):
    @abstractmethod
    def train(self,
              train_data: List[Datum],
              val_data: List[Datum],
              cache_featurizer: Optional[bool] = False) -> None:
        '''
        Performs training of model. The exact train implementations are model specific.
        :param train_data: List of train data
        :param val_data: List of validation data that can be used
        :param cache_featurizer: Whether or not to cache the model featurizer
        :return:
        '''
        pass

    @abstractmethod
    def predict(self, data: List[Datum]) -> np.array:
        '''
        Performs inference of model on collection of data. Returns an
        array of model predictions. This should only be called after the model
        has been trained.
        :param data: List of data to perform inference on
        :return: Array of predictions
        '''
        pass

    @abstractmethod
    def compute_metrics(self, eval_data: List[Datum], split: Optional[str] = None) -> Dict:
        '''
        Compute a set of model-specifc metrics on the provided set of data.
        :param eval_data: Data to compute metrics for
        :param split: Data split on which metrics are being computed
        :return: A dictionary mapping from the name of the metric to its value
        '''
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        '''
        Return the model-specific parameters such as number of hidden-units in the case
        of a neural network or number of trees for a random forest
        :return: Dictionary containing the model parameters
        '''
        pass
