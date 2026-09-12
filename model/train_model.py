"""
SpaceGuard AI - Machine Learning Model Training Pipeline
Trains high-performance ensemble classification models (Random Forest & Gradient Boosting)
for heavy rainfall prediction using combined meteorological and satellite features.
Adheres strictly to ML best practices with leakage-free splits and test evaluations.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Ensure root directory is on python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing.preprocess import DataPreprocessor
from model.evaluate_model import ModelEvaluator
from generate_sample_data import generate_dataset


def train_rainfall_model(
    data_path: str = None,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Complete ML Training Workflow:
    1. Load / generate dataset
    2. Clean and preprocess features
    3. Train-test split (80/20 stratified)
    4. Fit feature scaler on training split only
    5. Train Random Forest & Gradient Boosting models
    6. Select top performer and evaluate strictly on held-out test data
    7. Save model, scaler, feature metadata, and metrics.json
    """
    print("==================================================")
    print("🛰️ SPACEGUARD AI - MODEL TRAINING PIPELINE")
    print("==================================================")

    saved_model_dir = os.path.join(BASE_DIR, "model", "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    # 1. Load dataset
    if data_path is None:
        data_path = os.path.join(BASE_DIR, "data", "processed", "merged_dataset.csv")

    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Generating realistic multi-year dataset...")
        df = generate_dataset(num_records_per_location=550, seed=random_state)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
        if len(df) < 200:
            print(f"Dataset has {len(df)} rows. Generating expanded multi-year dataset (5,500 samples)...")
            df = generate_dataset(num_records_per_location=550, seed=random_state)
            df.to_csv(data_path, index=False)

    print(f"Loaded {len(df)} total meteorological and satellite records.")
    print(f"Heavy rainfall events in data: {df['heavy_rainfall_event'].sum()} ({df['heavy_rainfall_event'].mean():.1%})")

    # 2. Preprocess and Split
    preprocessor = DataPreprocessor(model_dir=saved_model_dir)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocessor.prepare_train_test(
        df=df, test_size=test_size, random_state=random_state
    )

    print(f"Training samples: {len(X_train_scaled)} | Test samples: {len(X_test_scaled)}")

    # 3. Model Training & Comparison
    print("\nTraining Ensemble Classifiers...")

    rf_model = RandomForestClassifier(
        n_estimators=160,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    gb_model = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=5,
        random_state=random_state
    )

    # 5-fold cross validation on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv, scoring="f1")
    gb_cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=cv, scoring="f1")

    print(f"Random Forest 5-Fold F1 Score:      {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std():.4f})")
    print(f"Gradient Boosting 5-Fold F1 Score:  {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std():.4f})")

    # Fit best model on entire training set
    if rf_cv_scores.mean() >= gb_cv_scores.mean():
        print("\nSelecting Random Forest Classifier as production model.")
        best_model = rf_model
    else:
        print("\nSelecting Gradient Boosting Classifier as production model.")
        best_model = gb_model

    best_model.fit(X_train_scaled, y_train)

    # 4. Strict Test-Set Evaluation
    print("\nEvaluating on Held-Out Test Set...")
    evaluator = ModelEvaluator(saved_model_dir=saved_model_dir)
    test_metrics = evaluator.evaluate(best_model, X_test_scaled, y_test, save_results=True)

    print(f"Test Accuracy:  {test_metrics['accuracy_pct']}%")
    print(f"Test Precision: {test_metrics['precision_pct']}%")
    print(f"Test Recall:    {test_metrics['recall_pct']}%")
    print(f"Test F1 Score:  {test_metrics['f1_pct']}%")
    print(f"Test ROC AUC:   {test_metrics['roc_auc']}")
    print(f"Confusion Matrix:\n {test_metrics['confusion_matrix']['raw_matrix']}")

    # 5. Save Model and Test Artifacts
    model_save_path = os.path.join(saved_model_dir, "rainfall_model.joblib")
    test_data_path = os.path.join(saved_model_dir, "test_data.joblib")

    joblib.dump(best_model, model_save_path)
    joblib.dump({"X_test": X_test_scaled, "y_test": y_test}, test_data_path)

    print(f"\n✅ Production model saved to: {model_save_path}")
    print("==================================================")
    print("TRAINING & EVALUATION COMPLETE!")
    print("==================================================")

    return best_model, test_metrics


if __name__ == "__main__":
    train_rainfall_model()
