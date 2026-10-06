import os
import joblib
#Set path to the downloaded model file
model_path = "./downloaded_model/artifacts/outputs/model.pkl"
#Load the model
model = joblib.load(model_path)
print("Model loaded successfully!", model)