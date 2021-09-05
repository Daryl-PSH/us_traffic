import pandas as pd

import datetime as datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Tuple

RAW_STATION_DATA_PATH = Path("data/raw/dot_traffic_stations_2015.txt.gz")
RAW_TRAFFIC_DATA_PATH = Path("data/raw/dot_traffic_2015.txt.gz")

def load_data(traffic_data_path: Path, station_data_path: Path) -> Tuple(pd.DataFrame):
    """
    Load the data from the traffic and station data path and return them as dataframe
    for further data cleaning

    Args:
        traffic_data_path (Path): File path to the raw traffic data
        station_data_path (Path): File path to the raw station data

    Returns:
        Tuple (DataFrame): Tuple containing (traffic_df, station_df)
    """
    traffic_df = pd.read_csv(traffic_data_path, compression="gzip")
    station_df = pd.read_csv(station_data_path, compression="gzip")

    return (traffic_df, station_df)

def drop_remapped_column(traffic_df: pd.DataFrame, station_df: pd.DataFrame) -> Tuple(pd.DataFrame):
    """
    Drop remapped columns that have been identified in the Data_Cleaning.ipynb before the state code filter
    of the dataframe

    Args:
        traffic_df (pd.DataFrame): DataFrame with traffic data
        station_df (pd.DataFrame): DataFrame with specific station data
    
    Returns:
        Tuple (DataFrame): Tuple containing (traffic_df, station_df)
    """
    station_columns_to_drop = ["algorithm_of_classification_name", "calibration_of_weighing_system_name",
                               "direction_of_travel", "functional_classification", "lane_of_travel_name",
                               "method_of_data_retrieval_name", "method_of_traffic_volume_counting_name",
                               "method_of_truck_weighing", "method_of_vehicle_classification_name",
                               "sample_type_of_traffic_volume", "sample_type_for_truck_weight_name",
                               "sample_type_for_vehicle_classification_name", "type_of_sensor"]

    traffic_df.drop(["restrictions"], inplace=True, axis=1)
    station_df.drop(station_columns_to_drop, axis=1, inplace=True)

    return (traffic_df, station_df)

def filter_by_state_code(traffic_df: pd.DataFrame, station_df: pd.DataFrame,
                         state_code: int) -> Tuple(pd.DataFrame):
    """
    Filter the dataframe by state code, keeping stations that belong to the specified code
    in the state_code argument.

    Refer to https://www.nrcs.usda.gov/wps/portal/nrcs/detail/?cid=nrcs143_013696
    for specific state code

    Args:
        traffic_df (pd.DataFrame): DataFrame with traffic data
        station_df (pd.DataFrame): DataFrame with specific station data
        state_code (int): State code of the state to predict the traffic volume

    Returns:
        Tuple (DataFrame): Tuple containing (traffic_df, station_df)
    """
    station_df = station_df[station_df["fips_state_code"] == state_code]
    traffic_df = traffic_df[traffic_df["fips_state_code"] == state_code]

    return traffic_df, station_df

def convert_established_year_to_actual_year(traffic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the year_of_data column to a year int format. Eg. 15 --> 2015

    Args:
        traffic_df (pd.DataFrame): DataFrame with traffic data

    Returns:
        pd.DataFrame: DataFrame with year_station_established column
    """
    traffic_df["year_station_established"].apply(lambda x: 2000 + x if x <= 15 else 1900 + x)

    return traffic_df

def remove_redundant_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any column that have cardinality of 1 as it does not increase the model's
    predictive power

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame: DataFrame with columns removed
    """
    for column in df.columns:
        if len(df[column].unique() == 1):
            df.drop(column, inplace=True, axis=1)

    return df