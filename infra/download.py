import os
import joblib

# Create an outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Save the trained model file directly into outputs
joblib.dump(model, 'outputs/model.pkl')