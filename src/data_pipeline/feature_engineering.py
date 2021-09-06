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

def create_peak_hour_traffic_volume_column(df: pd.DataFrame, rush_hour_type: str) -> pd.DataFrame:
    """
    Create a column called peak_hour_traffic_volume that consolidates the hours that comprises a peak hour

    Args:
        df (pd.DataFrame): Dataframe to create the new column
        rush_hour_type (str): Whether to predict am or pm rush hour (Only am or pm)

    Returns:
        pd.DataFrame: Modified dataframe with the new peak_hour_traffic_volume column
    """
    if rush_hour_type == "pm":
        peak_hour_columns = [
            "traffic_volume_counted_after_1400_to_1500", "traffic_volume_counted_after_1500_to_1600",
            "traffic_volume_counted_after_1600_to_1700", "traffic_volume_counted_after_1700_to_1800",
            "traffic_volume_counted_after_1800_to_1900"]
        df["peak_hour_traffic_volume"] = df[peak_hour_columns].sum(axis=1)
    
    elif rush_hour_type == "am":
        peak_hour_columns = [
            "traffic_volume_counted_after_0600_to_0700", "traffic_volume_counted_after_0700_to_0800",
            "traffic_volume_counted_after_0800_to_0900", "traffic_volume_counted_after_0900_to_1000"
        ]
        df["peak_hour_traffic_volume"] = df[peak_hour_columns].sum(axis=1)

    return df
