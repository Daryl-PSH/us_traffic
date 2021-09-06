import pandas as pd

import datetime as datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Tuple, List

RAW_STATION_DATA_PATH = Path("data/raw/dot_traffic_stations_2015.txt.gz")
RAW_TRAFFIC_DATA_PATH = Path("data/raw/dot_traffic_2015.txt.gz")

def load_data(traffic_data_path: Path, station_data_path: Path) -> Tuple[pd.DataFrame]:
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

def combine_data(traffic_df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine the traffic and station dataframe on their common column

    Args:
        traffic_df (pd.DataFrame): DataFrame with traffic volume information
        station_df (pd.DataFrame): DataFrame with station information

    Returns:
        pd.DataFrame
    """
    combined_df = traffic_df.merge(station_df,
                                   on=["station_id", "direction_of_travel_name",
                                       "functional_classification_name", "lane_of_travel",
                                       "year_of_data"])

    return combined_df

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

def columns_to_drop(columns_to_drop: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns specified in columns_to_drop list

    Args:
        columns_to_drop (List[str]): List of columns to drop
        df (pd.DataFrame): Dataframe to drop the specified columns

    Returns:
        pd.DataFrame: [description]
    """
    df.drop(columns_to_drop, inplace=True, axis=1)

    return df

def drop_future_volume_information(df: pd.DataFrame, rush_hour_type: str) -> pd.DataFrame:
    """
    Drop columns that will result in data leakage if not dropped as the model will have access to future data that it otherwise
    will not have access to
    Eg. Dropping any data after 7pm if predicting evening rush hour

    Args:
        df (pd.DataFrame): [description]
        rush_hour_type (str): Only two strings is accepted: am or pm

    Returns:
        pd.DataFrame: [description]
    """
    time_column_to_drop = []
    if rush_hour_type == "pm":
        for column in df.columns:
            hour = column.split("_")[-1]
            if int(hour) > 1900:
                time_column_to_drop.append(column)

    for column in time_column_to_drop:
        df.drop(column, inplace=True, axis=1)

    return df