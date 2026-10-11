import argparse
import glob
import json
import os
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def main(args):
    # read data[cite: 1]
    df = get_data(args.training_data)[cite: 1]

    # split data[cite: 1]
    X_train, X_test, y_train, y_test = split_data(df)[cite: 1]

    # train model[cite: 1]
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)[cite: 1]

    # evaluate model[cite: 1]
    metrics = eval_model(model, X_test, y_test)[cite: 1]

    # --- SAVE AND LOG MODEL ARTIFACTS ---
    # 1. Log model directly to MLflow[cite: 1]
    mlflow.sklearn.log_model(model, "model")[cite: 1]

    # 2. Save physical model.pkl to the 'outputs' directory[cite: 1]
    os.makedirs('outputs', exist_ok=True)[cite: 1]
    joblib.dump(model, 'outputs/model.pkl')[cite: 1]
    print("Saved model to outputs/model.pkl")[cite: 1]

    # 3. Automatically register the trained model to the Azure ML workspace registry
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model_name = "diabetes-model"
    print(f"Registering model from {model_uri} as '{registered_model_name}'...")
    mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    print("Model registration complete.")

    # persist metrics so the workflow can comment the actual values deterministically[cite: 1]
    if args.metrics_output:[cite: 1]
        save_metrics(metrics, args.metrics_output)[cite: 1]

def get_data(path):
    print("Reading data...")[cite: 1]

    if os.path.isdir(path):[cite: 1]
        csv_files = glob.glob(os.path.join(path, "*.csv"))[cite: 1]
        if not csv_files:[cite: 1]
            raise RuntimeError(f"No CSV files found in provided data path: {path}")[cite: 1]
        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)[cite: 1]
    else:
        df = pd.read_csv(path)[cite: 1]

    return df[cite: 1]

def split_data(df):
    print("Splitting data...")[cite: 1]
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values[cite: 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)[cite: 1]

    return X_train, X_test, y_train, y_test[cite: 1]

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    mlflow.log_param("Regularization rate", reg_rate)[cite: 1]
    print("Training model...")[cite: 1]
    model = LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)[cite: 1]

    return model[cite: 1]

def eval_model(model, X_test, y_test):
    y_hat = model.predict(X_test)[cite: 1]
    acc = np.average(y_hat == y_test)[cite: 1]
    print('Accuracy:', acc)[cite: 1]
    mlflow.log_metric("Accuracy", acc)[cite: 1]

    y_scores = model.predict_proba(X_test)[cite: 1]
    auc = roc_auc_score(y_test,y_scores[:,1])[cite: 1]
    print('AUC: ' + str(auc))[cite: 1]
    mlflow.log_metric("AUC", auc)[cite: 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])[cite: 1]
    fig = plt.figure(figsize=(6, 4))[cite: 1]
    plt.plot([0, 1], [0, 1], 'k--')[cite: 1]
    plt.plot(fpr, tpr)[cite: 1]
    plt.xlabel('False Positive Rate')[cite: 1]
    plt.ylabel('True Positive Rate')[cite: 1]
    plt.title('ROC Curve')[cite: 1]
    plt.savefig("ROC-Curve.png")[cite: 1]
    mlflow.log_artifact("ROC-Curve.png")[cite: 1]

    return {
        "accuracy": float(acc),
        "auc": float(auc),
    }[cite: 1]

def save_metrics(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)[cite: 1]
    metrics_path = os.path.join(output_dir, "metrics.json")[cite: 1]
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:[cite: 1]
        json.dump(metrics, metrics_file)[cite: 1]

    print(f"Saved metrics to {metrics_path}")[cite: 1]

def parse_args():
    parser = argparse.ArgumentParser()[cite: 1]

    parser.add_argument("--training_data", dest='training_data', type=str)[cite: 1]
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)[cite: 1]
    parser.add_argument("--metrics_output", dest='metrics_output', type=str, default=None)[cite: 1]

    args = parser.parse_args()[cite: 1]
    return args[cite: 1]

if __name__ == "__main__":
    print("\n\n")[cite: 1]
    print("*" * 60)[cite: 1]

    args = parse_args()[cite: 1]
    main(args)[cite: 1]

    print("*" * 60)[cite: 1]
    print("\n\n")[cite: 1]
