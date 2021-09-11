from abc import ABC, abstractmethod
from typing import Dict, Union, Literal

from ..data_pipeline.datapipeline import *
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

class Model(ABC):

    RAW_STATION_DATA_PATH = Path("data/raw/dot_traffic_stations_2015.txt.gz")
    RAW_TRAFFIC_DATA_PATH = Path("data/raw/dot_traffic_2015.txt.gz")
    
    def __init__(self, params):
        return self.build_model(params)

    def evaluate(self, y_true: np.array, y_pred: np.array,
                 metrics: Literal["mse", "rmse"] = "rmse"):
        """
        Evaluate the predictions using the specified metrics

        Args:
            y_true (np.array): Groundtruth value
            y_pred (np.array): Predicted value
            metrics (Literal[str]): Either mse or rmse (defaults rmse)
        """
        if metrics == "mse":
            return mean_squared_error(y_true, y_pred, squared=True)
        elif metrics == "rmse":
            return mean_squared_error(y_true, y_pred, squared=False)

    @abstractmethod
    def build_model(self, params: Dict[str, Union[int, str]]):
        """
        Build the model

        Args:
            params (Dict[str, Any]): Hyperparameters for the model

        Returns:
            model: Returns the model with the hyperparameter to be built
        """
        pass
    
    @abstractmethod
    def train(self, train_X, train_y):
        """
        Train the model

        Args:
            train_X (np.array): Features to be used for training
            train_y (np.array): Target label to be used for training

        """
        pass

    @abstractmethod
    def predict(self, data_X: np.array) -> np.array[int]:
        """
        Use model for prediction

        Args:
            data_X (np.array): Data to be used for prediction

        Returns:
            predictions (np.array): Prediction of the model 
        """
        pass

class RandomForest(Model):
    def build_model(self, params: Dict[str, Union[int, str]]):
        self.model = RandomForestRegressor(**params)

    def train(self, train_X, train_y):
        self.model.fit(train_X, train_y)

    def predict(self, data_X) -> (np.array[int]): 
        return np.round(self.model.predict(data_X))

class DecisionTree(Model):
    def build_model(self, params: Dict[str, Union[int, str]]):
        self.model = DecisionTreeRegressor(**params)

    def train(self, train_X, train_y):
        self.model.fit(train_X, train_y)

    def predict(self, data_X):
        return np.round(self.model.predict(data_X))




