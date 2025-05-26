# MLOps-Project
Repository contains end to end mlops project with detailed tutorial and code walkthroughs 

## Project components:

### 1. Data Loading
- Uses Yellow Taxi Trip data from NYC in parquet format
- Data loading implemented in [`load_data`](Yellow-Taxi-modeling-and-deployment/model_pipeline.py) function
- Loads trip data including pickup/dropoff locations, distances, and timestamps

### 2. Model Training Pipeline
- Feature engineering and preprocessing in [`preprocess_data`](Yellow-Taxi-modeling-and-deployment/model_pipeline.py)
- Data standardization using scikit-learn's StandardScaler
- Linear Regression model to predict trip duration
- Model evaluation using MSE metric
- Model and scaler artifacts saved using joblib

### 3. Model Inference Pipeline
- FastAPI web service for model serving
- REST API endpoint for trip duration predictions
- Takes pickup location, dropoff location, and trip distance as inputs
- Returns predicted trip duration

### 4. MLflow Tracking
- Experiment tracking and model versioning
- Logs model parameters, metrics, and artifacts
- Model registry for managing different versions
- UI available at http://127.0.0.1:5000

### 5. Workflow Orchestration
- Uses Prefect for pipeline orchestration
- Task dependencies and retries configured
- Handles data loading, preprocessing, training and evaluation
- Automated model artifact management

### Prefect Runs Image:

![Yellow-Taxi-modeling-and-deployment/Screenshot 2025-05-21 at 11.53.21 AM](https://github.com/avikumart/MLOps-Project/blob/main/Yellow-Taxi-modeling-and-deployment/Screenshot%202025-05-21%20at%2011.53.21%E2%80%AFAM.png)
