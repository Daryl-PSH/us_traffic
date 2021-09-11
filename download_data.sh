#!/usr/bin/env bash

STATION_DATA_PATH_OUTPUT="data/raw/dot_traffic_stations_2015.txt.gz"
STATION_DOWNLOAD_ID="1KvL-1CVGwx3TyH4gm7IEn-hw-5U9WriM"

TRAFFIC_DATA_PATH_OUTPUT="data/raw/dot_traffic_2015.txt.gz"
TRAFFIC_DOWNLOAD_ID="1o9vWVFROn2lnivu_rSO4UWGqZ4ZDHTrg"

gdown --id $STATION_DOWNLOAD_ID --output $STATION_DATA_PATH_OUTPUT
gdown --id $TRAFFIC_DOWNLOAD_ID --output $TRAFFIC_DATA_PATH_OUTPUT