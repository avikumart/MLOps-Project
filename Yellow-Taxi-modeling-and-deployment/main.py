import pandas as pd
import fastapi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import uvicorn

# create the functions to predict using developed model 

# creat the function to take the user input
class UserInput(BaseModel):
    PULocationID: int = Field(..., description="Pickup Location ID")
    DOLocationID: int = Field(..., description="Dropoff Location ID")
    trip_distance: float = Field(..., description="Trip Distance in miles")

    class Config:
        schema_extra = {
            "example": {
                "PULocationID": 1,
                "DOLocationID": 2,
                "trip_distance": 3.5
            }
        }


# create the function to load the model and scaler
def load_model(model_path: str, scaler_path: str):
    """
    Load the model from the file path
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict(model, scaler, data):
    """
    Predict the trip duration using the model and scaler
    """
    data = scaler.transform(data)
    prediction = model.predict(data)
    return prediction

app = FastAPI()

MODEL, SCALER = None, None

@app.on_event("startup")
async def load_ml_models():
    """Load ML models into memory"""
    global MODEL, SCALER
    MODEL, SCALER = load_model("model.joblib", "scaler.joblib")

# creat the prediction api endpoint for the user input
@app.post("/predict", response_model=dict)
async def predict_trip_duration(input_data: UserInput):
    """Predict the trip duration using the model and scaler"""
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict()])
        data = data[["PULocationID", "DOLocationID", "trip_distance"]]

        # Make prediction using global models
        prediction = predict(MODEL, SCALER, data)
        
        return {"trip_duration": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# create the function to run the app
