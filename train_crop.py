import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("crop_data.csv")

X = df[['N','P','K','ph','rainfall','temp']]
y = df['label']

# Train model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X, y)

# Save model
joblib.dump(clf, "crop_model.pkl")
print("✅ Model saved as crop_model.pkl")
