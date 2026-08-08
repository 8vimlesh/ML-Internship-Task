import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import re

app = FastAPI()

# Mount the static directory
app.mount("/public", StaticFiles(directory="public"), name="public")

# Define the request body schema
class PredictRequest(BaseModel):
    message: str

# Define the response schema
class PredictResponse(BaseModel):
    prediction: str
    confidence: float

# Global variables for model and vectorizer
model = None
vectorizer = None

@app.on_event("startup")
def load_model():
    global model, vectorizer
    if os.path.exists("model.joblib") and os.path.exists("vectorizer.joblib"):
        model = joblib.load("model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
    else:
        print("Warning: Model or vectorizer not found. Please run train.py first.")

@app.get("/")
def read_root():
    return FileResponse("public/index.html")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Ensure train.py has been run.")
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Clean the message identically to training
    cleaned_message = request.message.lower()
    cleaned_message = re.sub(r'[^a-z\s]', '', cleaned_message)
    cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()

    # Vectorize the input message
    vec_message = vectorizer.transform([cleaned_message])
    
    # Predict the class (spam or ham)
    prediction = model.predict(vec_message)[0]
    
    # Get the prediction probabilities
    proba = model.predict_proba(vec_message)[0]
    confidence = max(proba)
    
    return PredictResponse(prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
