
import joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model.joblib"

# Load trained model
model = joblib.load(MODEL_PATH)

# Create FastAPI application
app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict an expected score from study hours",
    version="1.0.0"
)

# Input validation
class StudentInput(BaseModel):
    study_hours: float = Field(ge=0, le=10)

@app.get("/")
def home():
    return {
        "message": "Student Performance API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(data: StudentInput):
    prediction = model.predict([[data.study_hours]])[0]

    return {
        "study_hours": data.study_hours,
        "predicted_score": round(float(prediction), 2)
    }
