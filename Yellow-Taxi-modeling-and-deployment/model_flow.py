from prefect import flow, task
import model_pipeline 
import pandas as pd
import os
import joblib

@task(retries=3, retry_delay_seconds=5)
def load_data(file_path: str) -> pd.DataFrame:
    return model_pipeline.load_data(file_path)

@task(retries=3, retry_delay_seconds=5)
def preprocess_data(data: pd.DataFrame) -> tuple:
    return model_pipeline.preprocess_data(data)

@task(retries=3, retry_delay_seconds=5)
def standardize_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    return model_pipeline.standardize_data(X_train,X_test)

@task(retries=3, retry_delay_seconds=5)
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> joblib:
    return model_pipeline.train_model(X_train, y_train)

@task
def evaluate_model(model: joblib, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    return model_pipeline.evaluate_model(model, X_test, y_test)

@task
def save_model(model: joblib, file_path: str) -> None:
    model_pipeline.save_model(model, file_path)

@task
def save_scaler(scaler:joblib, file_path: str) -> None:
    """
    Save the scaler
    """
    joblib.dump(scaler, file_path)

# create the python function to run the flow
@flow
def run_flow(file_path: str) -> None:
    """
    Run the flow
    """
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train, X_test, scaler = standardize_data(X_train, X_test)
    model = train_model(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    save_model(model, "model.joblib")
    save_scaler(scaler, "scaler.joblib")
    print("model and scaler are saved")

if __name__ == "__main__":
    run_flow("/Users/avikumart/Documents/GitHub/MLOps-Project/Data/yellow_tripdata_2025-01.parquet")