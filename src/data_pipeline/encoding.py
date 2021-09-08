import pandas as pd

def load_data(interim_data_path: str) -> pd.DataFrame:
    """
    Load data from the interim data path after preprocessing is done

    Args:
        interim_data_path (str): Data path to interim data

    Returns:
        pd.DataFrame: Preprocessed data
    """
    preprocess_df = pd.read_csv(interim_data_path)
    preprocess_df["date"] = pd.to_datetime(preprocess_df["date"])

    return preprocess_df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the object type in DataFrame to categorical to be used for modelling
    downstream

    Args:
        df (pd.DataFrame)

    Returns:
        pd.DataFrame: [description]
    """
    object_list = []

    for column in df.columns:
        if df[column].dtype == "object":
            object_list.append(column)
        
    df[object_list].astype("category")

    return df