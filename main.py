from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import requests
import json
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Farmer Voice Assistant API")

# Enable CORS (so frontend can connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your website URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load crop model
model = joblib.load("crop_model.pkl")

# Load schemes
with open("schemes_db.json", "r", encoding="utf-8") as f:
    SCHEMES = json.load(f)

# Weather API key
OPENWEATHER_KEY = os.environ.get("OPENWEATHER_KEY", "YOUR_KEY_HERE")



# --------- Request Models ---------
class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    ph: float
    rainfall: float
    temp: float

class ChatRequest(BaseModel):
    text: str
    lang: str = "hi"

# --------- Endpoints ---------
@app.post("/predict_crop")
def predict_crop(req: CropRequest):
    arr = np.array([[req.N, req.P, req.K, req.ph, req.rainfall, req.temp]])
    crop = model.predict(arr)[0]
    return {"recommended_crop": crop}

@app.get("/weather/{lat}/{lon}")
def get_weather(lat: float, lon: float):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
    r = requests.get(url)
    return r.json()

@app.get("/schemes")
def list_schemes():
    return {"schemes": SCHEMES}

@app.post("/chat")
def chat(req: ChatRequest):
    # Simple rule-based chatbot (can replace with AI later)
    if "मौसम" in req.text or "weather" in req.text.lower():
        return {"reply": "कल बारिश होगी और तापमान 28°C रहेगा।"}
    elif "फसल" in req.text or "crop" in req.text.lower():
        return {"reply": "आपके खेत की स्थिति के अनुसार धान (Rice) सबसे अच्छी फसल होगी।"}
    elif "योजना" in req.text or "scheme" in req.text.lower():
        return {"reply": "प्रधानमंत्री फसल बीमा योजना उपलब्ध है।"}
    else:
        return {"reply": "मैं आपकी मदद के लिए तैयार हूँ। कृपया फसल, मौसम या योजना से संबंधित पूछें।"}
