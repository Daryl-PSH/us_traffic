import pandas as pd

import datetime as datetime
from dateutil.relativedelta import relativedelta

def create_max_volume_column(traffic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the max_volume_column for the traffic dataframe which keep tracks of the total
    daily traffic volume

    Args:
        traffic_df (pd.DataFrame): DataFrame with traffic volume information

    Returns:
        pd.DataFrame: DataFrame with the newly created volume column
    """
    volume_columns = [column for column in traffic_df.columns if column.startswith("traffic_volume")]
    traffic_df["total_volume"] = traffic_df["volume_columns"].sum(axis=1)

    return traffic_df

def create_years_of_operation_column(traffic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the years of operation column which indicates how long the station has been running

    Args:
        traffic_df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """

    traffic_df["year_of_data"] = traffic_df["year_of_data"] + 2000
    traffic_df["year_of_service"] = traffic_df["year_of_data"] - traffic_df["year_station_established"]

    return traffic_df
