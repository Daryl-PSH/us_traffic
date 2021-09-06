from feature_engineering import *
from preprocess_data import *

from pathlib import Path

RAW_STATION_DATA_PATH = Path("data/raw/dot_traffic_stations_2015.txt.gz")
RAW_TRAFFIC_DATA_PATH = Path("data/raw/dot_traffic_2015.txt.gz")
STATE_CODE = 36

def preprocess_data(traffic_data_path: Path, station_data_path: Path):

    traffic_df, station_df = load_data(traffic_data_path, station_data_path)
    traffic_df, station_df = drop_remapped_column(traffic_df, station_df)
    traffic_df, station_df = filter_by_state_code(traffic_df, station_df, state_code=STATE_CODE)

    # Combine dataframe
    combined_df = combine_data(traffic_df, station_df)

    # Feature engineering
    combined_df = convert_established_year_to_actual_year(combined_df)
    combined_df = create_years_of_operation_column(combined_df)

    combined_df.to_csv("data/interim/cleaned_data.csv")


