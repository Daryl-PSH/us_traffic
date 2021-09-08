import pandas as pd

from typing import Tuple

def convert_and_sort_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert and sort the date column to be used for train test split

    Args:
        df (pd.DataFrame): DataFrame with date

    Returns:
        pd.DataFrame
    """
    df.sort_values("date", inplace=True)

    return df

def train_val_test_split(df: pd.DataFrame, train_ratio: float,
                         validation_ratio: float, test_ratio: float) -> Tuple[pd.DataFrame]:
    """
    Split the data in training, validation and test set. Where the latest  data will be in the test set,
    followed by val and the model will be trained on the earlier set (train)

    Args:
        df (pd.DataFrame):

    Returns:
        Tuple[pd.DataFrame]: Tuple of train/validation/test
    """
    NUM_MONTHS = 12
     
    if train_ratio + validation_ratio + test_ratio != 1:
        return ValueError("Split ratios must add up to 1!")

    train_month = round(NUM_MONTHS * train_ratio)
    val_month = round(NUM_MONTHS * validation_ratio)
    test_month = round(NUM_MONTHS * test_ratio)

    train_set = df.loc[df["month_of_data"] <= train_month].copy(deep=True)
    val_set = df.loc[(df["month_of_data"] > train_month) &
                     (df["month_of_data"] <= NUM_MONTHS - test_month)].copy(deep=True)
    test_set = df.loc[df["month_of_data"] > train_month + val_month ].copy(deep=True)

    return train_set, val_set, test_set


