from typing import Dict, Union
from src.models.model import *

class ModelFactory:
    def create_model(model_type: str, params: Dict[str, Union[int, str]]):
        if model_type == "random_forest":
            return RandomForest(params)

        elif model_type == "decision_tree":
            return DecisionTree(params)
