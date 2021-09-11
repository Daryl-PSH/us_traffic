import pandas as pd
from src.data_pipeline.feature_engineering import *

from dateutil.relativedelta import relativedelta
from pathlib import Path
from typing import Tuple, List

RAW_STATION_DATA_PATH = Path("data/raw/dot_traffic_stations_2015.txt.gz")
RAW_TRAFFIC_DATA_PATH = Path("data/raw/dot_traffic_2015.txt.gz")
STATE_CODE = 36

def preprocess_data(traffic_data_path: Path, station_data_path: Path) -> pd.DataFrame:
    """
    Preprocess the data to be used for encoding, first step of the data pipeline

    Args:
        traffic_data_path (Path): Filepath to the traffic csv
        station_data_path (Path): Filepath to the station csv

    Returns:
        pd.DataFrame:
    """
    traffic_df, station_df = load_data(traffic_data_path, station_data_path)

    traffic_df, station_df = drop_remapped_column(traffic_df, station_df)
    traffic_df, station_df = filter_by_state_code(traffic_df, station_df, state_code=STATE_CODE)
    station_df = clean_sample_type_for_vehicle_classification_column(station_df)

    # Combine dataframe
    combined_df = combine_data(traffic_df, station_df)

    # Feature engineering
    combined_df = convert_established_year_to_actual_year(combined_df)
    combined_df = create_years_of_operation_column(combined_df)
    combined_df = create_peak_hour_traffic_volume_column(combined_df, rush_hour_type="pm")

    # Final column drop before encoding features
    combined_df = remove_redundant_column(combined_df)
    combined_df = drop_future_volume_information(combined_df, "pm")

    columns_to_drop = ["year_station_established", "station_location",
                       "previous_station_id", "latitude", "longitude",
                       "method_of_truck_weighing_name", "fips_county_code",
                       "direction_of_travel", "station_id", "hpms_sample_identifier",
                       "algorithm_of_vehicle_classification", "functional_classification_name"]

    combined_df = drop_columns(columns_to_drop, combined_df)
    combined_df = drop_na_columns(combined_df)

    combined_df.to_csv("data/interim/cleaned_data.csv", index=False)

    return combined_df

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

def drop_remapped_column(traffic_df: pd.DataFrame, station_df: pd.DataFrame) -> Tuple[pd.DataFrame]:
    """
    Drop remapped columns that have been identified in the Data_Cleaning.ipynb before the state code filter
    of the dataframe

    Args:
        traffic_df (pd.DataFrame): DataFrame with traffic data
        station_df (pd.DataFrame): DataFrame with specific station data
    
    Returns:
        Tuple (DataFrame): Tuple containing (traffic_df, station_df)
    """
    station_columns_to_drop = ["algorithm_of_vehicle_classification_name", "calibration_of_weighing_system_name",
                               "direction_of_travel", "functional_classification", "lane_of_travel_name",
                               "method_of_data_retrieval_name", "method_of_traffic_volume_counting_name",
                               "method_of_truck_weighing", "method_of_vehicle_classification_name",
                               "sample_type_for_traffic_volume", "sample_type_for_truck_weight_name",
                               "sample_type_for_vehicle_classification_name", "type_of_sensor"]

    traffic_df.drop("restrictions", inplace=True, axis=1)
    station_df.drop(station_columns_to_drop, axis=1, inplace=True)

    return (traffic_df, station_df)

def filter_by_state_code(traffic_df: pd.DataFrame, station_df: pd.DataFrame,
                         state_code: int) -> Tuple[pd.DataFrame]:
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
    traffic_df["year_station_established"] = traffic_df["year_station_established"].apply(lambda x: 2000 + x if x <= 15 else 1900 + x)

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
        if len(df[column].unique()) == 1:
            df.drop(column, inplace=True, axis=1)

    return df

def drop_columns(columns_to_drop: List[str], df: pd.DataFrame) -> pd.DataFrame:
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
            if column.startswith("traffic_volume"):
                hour = column.split("_")[-1]
                if int(hour) > 1900:
                    time_column_to_drop.append(column)

    for column in time_column_to_drop:
        df.drop(column, inplace=True, axis=1)

    return df

def drop_na_columns(df: pd.DataFrame, threshold: float=0.5) -> pd.DataFrame:
    """
    Drop columns with more than 50% missing values

    Args:
        df (pd.DataFrame): Dataframe to drop columns
        threshold (float): Threshold before the column is dropped

    Returns:
        pd.DataFrame: Dataframe with dropped columns
    """
    thresh = df.shape[0] * threshold
    df = df.dropna(axis=1, thresh=thresh)

    return df

def clean_sample_type_for_vehicle_classification_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use the mapping of:
    N and nans to 0
    Y, 2, T, H to 1
    Where 0 = Station not used for Heavy Vehicle Travel Information System
    and 1 = Station used for Heavy Vehicle Travel Information System

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
    df["sample_type_for_vehicle_classification"].replace(["H", "Y", "2", "T"], "1",inplace=True)
    df["sample_type_for_vehicle_classification"].replace("N", "0",inplace=True)
    df["sample_type_for_vehicle_classification"].fillna("0", inplace=True)
    df["sample_type_for_vehicle_classification"] = pd.to_numeric(df["sample_type_for_vehicle_classification"])

    return df
