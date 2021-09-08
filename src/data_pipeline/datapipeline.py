import os
from pathlib import Path
import logging

from utils import train_val_test_split
from feature_engineering import *
from preprocess_data import *
from encoding import encode_categorical
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Data Pipeline")

RAW_STATION_DATA_PATH = Path("data/raw/dot_traffic_stations_2015.txt.gz")
RAW_TRAFFIC_DATA_PATH = Path("data/raw/dot_traffic_2015.txt.gz")

def run_pipeline(traffic_data_path: Path, station_data_path: Path):
    TRAIN_FILE_PATH = "data/processed/train.csv"
    VAL_FILE_PATH = "data/processed/val.csv"
    TEST_FILE_PATH = "data/processed/test.csv"

    if not os.path.isfile(TRAIN_FILE_PATH):

        logger.info("Preprocessing Data")
        combined_df = preprocess_data(RAW_TRAFFIC_DATA_PATH, RAW_STATION_DATA_PATH)

        logger.log("Encoding data")
        combined_df = encode_categorical(combined_df)

        logger.log("Splitting data")
        train, val, test = train_val_test_split(combined_df, 0.8, 0.1, 0.1)

        train.to_csv(TRAIN_FILE_PATH, header=False)
        val.to_csv(VAL_FILE_PATH, header=False)
        test.to_csv(TEST_FILE_PATH, header=False)
    
    else:
        logger.info("Train, val, test csv found, reading them")
        train = pd.read_csv(TRAIN_FILE_PATH)
        val = pd.read_csv(VAL_FILE_PATH)
        test = pd.read_csv(TEST_FILE_PATH)


    train_X = train.drop("peak_hour_traffic_volume", axis=1).copy()
    train_y = train["peak_hour_traffic_volume"].copy()

    val_X = val.drop("peak_hour_traffic_volume", axis=1).copy()
    val_y = val["peak_hour_traffic_volume"].copy()

    test_X = test.drop("peak_hour_traffic_volume", axis=1)
    test_y = test["peak_hour_traffic_volume"].copy()

    logger.info("Encoding and scaling data")
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    num_columns = train_X.select_dtypes(include=np.number).columns.tolist()
    cat_columns = train_X.select_dtypes(include="object").columns.tolist()

    train_scaled_columns = scaler.fit_transform(train[num_columns])
    train_encoded_columns = encoder.fit_transform(train[cat_columns])
    train_processed = np.concatenate([train_scaled_columns, train_encoded_columns], axis=1)

    val_scaled_columns = scaler.transform(val[num_columns])
    val_encoded_columns = encoder.transform(val[cat_columns])
    val_processed = np.concatenate([val_scaled_columns, val_encoded_columns], axis=1)

    test_scaled_columns = scaler.transform(test[num_columns])
    test_encoded_columns = encoder.transform(test[cat_columns])
    test_processed = np.concatenate([test_scaled_columns, test_encoded_columns], axis=1)

    return {
        "train": [train_processed, train_y],
        "val": [val_processed, val_y],
        "test": [test_processed, test_y]
    }










