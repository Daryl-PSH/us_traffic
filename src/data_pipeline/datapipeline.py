from os import remove
from feature_engineering import *
from preprocess_data import *

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Data Pipeline")

RAW_STATION_DATA_PATH = Path("data/raw/dot_traffic_stations_2015.txt.gz")
RAW_TRAFFIC_DATA_PATH = Path("data/raw/dot_traffic_2015.txt.gz")
STATE_CODE = 36

def preprocess_data(traffic_data_path: Path, station_data_path: Path):

    logger.info("Loading data...")
    traffic_df, station_df = load_data(traffic_data_path, station_data_path)

    logger.info("Cleaning Data")
    traffic_df, station_df = drop_remapped_column(traffic_df, station_df)
    traffic_df, station_df = filter_by_state_code(traffic_df, station_df, state_code=STATE_CODE)
    station_df = clean_sample_type_for_vehicle_classification_column(station_df)

    # Combine dataframe
    logger.info("Combining dataframe")
    combined_df = combine_data(traffic_df, station_df)

    # Feature engineering
    logger.info("Creating new features")
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

