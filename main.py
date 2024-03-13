from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = FastAPI()

# Load the trained model
loaded_model = joblib.load('model.pkl')
# Load the vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Define Pydantic BaseModel for request body
class SymptomInput(BaseModel):
    symptoms: str

# Define Pydantic BaseModel for response body
class DiseasePrediction(BaseModel):
    disease: str

# Endpoint for predicting diseases based on symptoms
@app.post("/predict_disease/", response_model=DiseasePrediction)
async def predict_disease(symptoms_input: SymptomInput):
    try:
        # Preprocess the input symptoms
        input_text = [symptoms_input.symptoms]
        input_vectorized = vectorizer.transform(input_text)

        # Make predictions using the trained model
        predicted_disease = loaded_model.predict(input_vectorized)

        return {"disease": predicted_disease[0]}
    except Exception as e:
        return {"error": str(e)}
