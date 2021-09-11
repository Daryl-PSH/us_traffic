from src.data_pipeline import datapipeline
from src.models.model_factory import ModelFactory

import yaml
from pathlib import Path
from typing import Dict, Union
import logging
import pickle

RAW_STATION_DATA_PATH = Path("data/raw/dot_traffic_stations_2015.txt.gz")
RAW_TRAFFIC_DATA_PATH = Path("data/raw/dot_traffic_2015.txt.gz")
CONF_PATH = Path("conf/model.yaml")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Training")

def experiment():

    logger.info("Loading Configurations")
    conf = load_conf(CONF_PATH)
    model_to_be_used = conf["model"]
    model_params = conf[model_to_be_used]

    data = datapipeline.run_pipeline(RAW_TRAFFIC_DATA_PATH, RAW_STATION_DATA_PATH)

    train_X, train_y = data["train"][0], data["train"][1]
    val_X, val_y = data["val"][0], data["val"][1]
    test_X, test_y = data["test"][0], data["test"][1]

    logger.info("Building model")
    model_factory = ModelFactory()
    model = model_factory.create_model(model_to_be_used, model_params)

    logger.info("Training model, might take a while")
    model.train(train_X, train_y)

    logger.info("Performing predictions + evaluations")
    train_predictions = model.predict(train_X)
    val_predictions = model.predict(val_X)
    test_predictions = model.predict(test_X)

    train_score = model.evaluate(train_y, train_predictions, metrics="rmse")
    val_score = model.evaluate(val_y, val_predictions, metrics="rmse")
    test_score  = model.evaluate(test_y, test_predictions, metrics="rmse")

    logger.info("Saving and logging results")
    model.save_model()
    model.log_results(train_score, val_score, test_score)


def load_conf(conf_path: Path) -> Dict[str, Union[str, int]]:
    """
    Load configuration file to be used
    """
    with open(conf_path, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return conf

if __name__ == "__main__":
    experiment()
