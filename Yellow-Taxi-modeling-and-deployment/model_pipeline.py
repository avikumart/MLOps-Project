import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os

# create the python function to load the data from the parquet file
def load_data(file_path):
    """
    Load the data from the parquet file
    """
    data = pd.read_parquet(file_path)
    return data

# create the python function to preprocess the data by creating the features and target variable, target variable is the trip duration
def preprocess_data(data):
    """
    preprocess the data by creating the features and target variable
    """
    data["duration"] = (data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]).dt.total_seconds() / 60

    # create the features and target variable
    X = data[["PULocationID", "DOLocationID", "trip_distance"]]
    y = data["duration"]
    # create the train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# create the python function to standardize the data
def standardize_data(X_train, X_test):
    """
    Standardize the data
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

# create the python function to train the model
def train_model(X_train, y_train):
    """
    Train the model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
# create the python function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# create the python function to save the model
def save_model(model, file_path):
    """
    Save the model
    """
    joblib.dump(model, file_path)
