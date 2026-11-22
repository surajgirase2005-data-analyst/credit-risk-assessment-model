from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

ml_model: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["model"] = joblib.load(BASE_DIR / "credit_risk_model_final.pkl")
    ml_model["threshold"] = float(joblib.load(BASE_DIR / "best_threshold_final.pkl"))
    yield
    ml_model.clear()


app = FastAPI(title="Credit Risk Assessment System", version="1.0.0", lifespan=lifespan)


class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, le=100)
    person_income: float = Field(..., gt=0)
    person_home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
    person_emp_length: float = Field(..., ge=0, le=60)
    loan_intent: Literal[
        "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
        "HOMEIMPROVEMENT", "DEBTCONSOLIDATION",
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    loan_amnt: float = Field(..., gt=0)
    loan_int_rate: float = Field(..., ge=0, le=100)
    loan_percent_income: float = Field(..., ge=0, le=1)
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: int = Field(..., ge=0, le=60)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in ml_model}


@app.post("/predict")
def predict(data: LoanApplication):
    input_df = pd.DataFrame([data.model_dump()])

    probability = float(ml_model["model"].predict_proba(input_df)[:, 1][0])
    # 3. Clip ONLY for displaying percentage
    display_probability = np.clip(
        probability, 0.001, 0.999
    )
    threshold = ml_model["threshold"]
    prediction = int(probability >= threshold)

    return {
        "default_probability": round(display_probability, 3),
        "default_prediction": prediction,
        "threshold": 0.2032,
        "Result": "High Risk" if prediction == 1 else "Low Risk",
    }


# Mounted last so API routes take precedence over static files.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
