import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Replace with your CSV file path
data = pd.read_csv("Crop_recommendation.csv")  

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = data[FEATURE_COLS]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))

# Save model
pickle.dump(model, open("crop_model.pkl", "wb"))
