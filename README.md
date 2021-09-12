US Traffic Prediction
==============================

## Problem Statement

Predict traffic volume during rush hour in New York. (Rush hour is defined to be from 3pm to 7pm) to be used for traffic planning by relevant entities

--------
## Project Organization


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
Architecture
==============================
Usage
==============================
## 1. Setting Up

1. Create a virtual environment of your choice (Anaconda or virtualenv) and run the following command in the root directory (in your virtual environment) to install the required dependencies

```
pip install -r requirements.txt
```

2. As the data is relatively large, run the shell script in the root directory called "download_data.sh" to download the data into the ```data/raw/``` folder

```
sh download_data.sh
```

3. Check if you have two gz files in the ```data/raw/``` folder called ```dot_traffic_2015.txt.gz``` and ```dot_traffic_stations_2015.txt.gz```

## 2. Configuration

1. To train the model, you will only have to modify the values and hyperparameters in ```conf/model.yaml``` (Currently only ```decision_tree``` and ```random_forest```)

## 3. Training

1. Run the following command in the root directory to train the model and it will commence training using the parameters that was set in ```conf/model.yaml``` from earlier.

```
python -m src.train
```

2. A time stamped folder will be created in the ```models``` folder along with the type of model that is being trained. Within the specific folder itself, there will be a joblib file that stores the model and also a yaml file containing the evaluation results of the model that was trained (Defaults to RMSE) 



--------
Improvements
==============================
1. Add in more models to be used in the config file
2. Implement Docker so that running of the scripts will be system agnostic
3. Experiment with heavier architectures such as Neural Networks
4. Implement automation of hyperparameter optimization techiniques such as GridSearch or RandomSearch.


References
==============================
1. Dataset: https://www.kaggle.com/jboysen/us-traffic-2015
