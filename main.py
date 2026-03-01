# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load the saved model
try:
    diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
except FileNotFoundError:
    raise RuntimeError("diabetes_model.pkl not found. Ensure it's committed to your repo.")

@app.post("/diabetes_prediction")
def diabetes_pred(input_parameters: ModelInput):
    try:
        # Extract values directly from Pydantic model (cleaner than JSON parsing)
        input_list = [
            input_parameters.Pregnancies,
            input_parameters.Glucose,
            input_parameters.BloodPressure,
            input_parameters.SkinThickness,
            input_parameters.Insulin,
            input_parameters.BMI,
            input_parameters.DiabetesPedigreeFunction,
            input_parameters.Age
        ]
        
        prediction = diabetes_model.predict([input_list])
        
        result = "You are not Diabetic" if prediction[0] == 0 else "You are Diabetic"
        return {"prediction": result, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Service is running"}
