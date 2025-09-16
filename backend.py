from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("crop_model.pkl", "rb"))
class_names = model.classes_


app = FastAPI()
# Crop info (you can expand this)
crop_info = {
    "rice": {"image": "", "tips": "Grows well in clayey soil with standing water."},
    "maize": {"image": "", "tips": "Needs warm climate, sandy loam soil."},
    "wheat": {"image": "", "tips": "Best in loamy soil, cool climate."}
}



# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class FarmerInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/recommend")
def recommend_crop(data: FarmerInput):
    features = np.array([[data.N, data.P, data.K, data.temperature,
                          data.humidity, data.ph, data.rainfall]])
    probs = model.predict_proba(features)[0]

    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = []
    for i in top3_idx:
        crop_name = class_names[i]
        info = crop_info.get(crop_name.lower(), {"image": "", "tips": "No tips available"})
        top3.append({
            "crop": crop_name,
            "confidence": round(probs[i]*100, 2),
            "image": info["image"],
            "tips": info["tips"]
        })
    return {"recommendations": top3}
