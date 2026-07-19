"""
modelling.py - Kriteria 2 Basic Level
Membangun model machine learning dengan MLflow autolog
"""

import pandas as pd
import numpy as np
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PARSE ARGUMENTS (untuk MLProject)
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=5)
parser.add_argument('--random_state', type=int, default=42)
args = parser.parse_args()

print("="*60)
print("📊 KRITERIA 2: MEMBANGUN MODEL ML")
print("="*60)
print(f"\n📌 PARAMETER:")
print(f"   n_estimators: {args.n_estimators}")
print(f"   max_depth: {args.max_depth}")
print(f"   random_state: {args.random_state}")

# ============================================
# 1. LOAD DATA PREPROCESSING
# ============================================
print("\n1. LOAD DATA...")

# Load data hasil preprocessing (path berbeda untuk MLProject)
df = pd.read_csv('namadataset_preprocessing/data_preprocessed.csv')

print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print("\n5 Data Pertama:")
print(df.head())

# ============================================
# 2. SPLIT DATA
# ============================================
print("\n2. SPLIT DATA...")

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ X_train: {X_train.shape}")
print(f"✅ X_test: {X_test.shape}")
print(f"✅ y_train: {y_train.shape}")
print(f"✅ y_test: {y_test.shape}")

# ============================================
# 3. TRAINING MODEL DENGAN MLflow AUTOLOG
# ============================================
print("\n3. TRAINING MODEL DENGAN MLFLOW...")

# Set experiment name
mlflow.set_experiment("Titanic_Survival_Prediction")

# Start MLflow run
with mlflow.start_run(run_name=f"RF_{args.n_estimators}_{args.max_depth}"):
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    # Inisialisasi model dengan parameter dari argparse
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    
    # Training
    print(f"   Training Random Forest (n={args.n_estimators}, depth={args.max_depth})...")
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    # Log metrics tambahan
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    
    print(f"\n📊 RESULTS:")
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Precision: {test_precision:.4f}")
    print(f"   Test Recall: {test_recall:.4f}")
    print(f"   Test F1 Score: {test_f1:.4f}")
    
    # Confusion Matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    print("\n✅ Training selesai! MLflow telah mencatat semua parameter dan metrics.")
    print(f"✅ Model disimpan di: {mlflow.get_artifact_uri()}")

print("\n" + "="*60)
print("✅ MODELLING COMPLETE!")
print("="*60)