US Traffic Prediction
==============================
## Problem Statement

Predict traffic volume during rush hour in New York. (Rush hour is defined to be from 3pm to 7pm) to be used for traffic planning by relevant entities

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained models and evaluation results
    │
    ├── conf               
    │   └── model.yaml     <- Configuration file for model training
    │
    ├── notebooks          <- Jupyter notebooks
    ├── environment.yml    <- To replicate in a conda environment
    ├── requirements.txt   <- The requirements file for reproducing the environment
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── train.py       <- Script to train and save model
    │   ├── models         <- Modules to build models for experiments
    │   │   ├── model_factory.py
    │   │   └── model.py
    │   └── data_pipeline  <- Scripts to preprocess and clean the data
    │       └── utils.py
    │       └── datapipeline.py
    │       └── encoding.py
    │       └── feature_engineering.py
    │       └── preprocess_data.py
    └── download_data.sh   <- Shell script for downloading raw data


--------
