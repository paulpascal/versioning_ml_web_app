"""
Test suite for model training functionality using student performance data.

This test suite verifies that our machine learning models can be trained correctly
on real-world student performance data. It tests:
1. Data loading and preparation
2. Model training with different algorithms
3. Model performance metrics
4. Feature importance analysis
5. Model saving functionality

The tests use the student_performance.csv dataset which contains:
- Features: study_hours, previous_score, attendance_rate
- Target: pass (binary classification)
"""

from datetime import datetime
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from app.utils.model_handler import ModelHandler

# Get the directory where this test file is located
TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def get_timestamped_name(base_name):
    """Generate a timestamped name for files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}"


@pytest.fixture
def student_data():
    """
    Fixture to load and prepare student performance data.

    Returns:
        dict: Contains:
            - X: Feature matrix (study_hours, previous_score, attendance_rate)
            - y: Target vector (pass/fail)
            - features: List of feature names
            - target: Target column name
            - df: Original DataFrame
    """
    # Load the dataset from the tests directory
    csv_path = os.path.join(TEST_DIR, "student_performance.csv")
    df = pd.read_csv(csv_path)

    # Prepare features and target
    features = ["study_hours", "previous_score", "attendance_rate"]
    target = "pass"

    # Split the data
    X = df[features].values
    y = df[target].values

    return {"X": X, "y": y, "features": features, "target": target, "df": df}


@pytest.fixture
def model_config(student_data):
    """
    Fixture to provide model configuration based on student data.

    Returns:
        dict: Model configuration including:
            - features: List of feature names
            - target: Target column name
            - normalize: Whether to normalize features
            - train_size: Proportion of data to use for training
    """
    return {
        "features": student_data["features"],
        "target": student_data["target"],
        "normalize": True,
        "train_size": 0.8,
    }


@pytest.mark.data_validation
def test_data_loading(student_data):
    """
    Test that the student performance data is loaded correctly.

    Verifies:
    1. Data has rows (not empty)
    2. Features and target are properly loaded
    3. Correct number of features
    4. Correct target column name
    """
    print("\n=== Data Loading Test ===")
    print(f"Number of samples: {student_data['X'].shape[0]}")
    print(f"Number of features: {len(student_data['features'])}")
    print(f"Target column: {student_data['target']}")

    assert student_data["X"].shape[0] > 0
    assert student_data["y"].shape[0] > 0
    assert len(student_data["features"]) == 3
    assert student_data["target"] == "pass"


@pytest.mark.model_training
def test_random_forest_classifier(student_data, model_config):
    """
    Test Random Forest classifier on student performance data.

    Verifies:
    1. Model can be trained successfully
    2. Training and test accuracy are reasonable (> 0.5)
    3. Confusion matrix is properly generated
    4. Feature importance is calculated and valid
    5. All metrics are within expected ranges
    """
    print("\n=== Random Forest Classifier Test ===")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        student_data["X"], student_data["y"], test_size=0.2, random_state=42
    )

    # Initialize model handler
    model_handler = ModelHandler(
        model_type="random_forest_classifier",
        features=model_config["features"],
        target=model_config["target"],
        normalize=model_config["normalize"],
    )

    # Train model
    success = model_handler.train(X_train, X_test, y_train, y_test)
    assert success, "Model training failed"

    # Verify model was trained
    assert model_handler.model is not None
    assert model_handler.results is not None

    # Print results
    print("\nModel Results:")
    print(f"Training Accuracy: {model_handler.results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {model_handler.results['test_accuracy']:.4f}")

    print("\nConfusion Matrix:")
    cm = model_handler.results["confusion_matrix"]
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")

    print("\nFeature Importance:")
    for feature in model_handler.results["feature_importance"]:
        print(f"{feature['feature']}: {feature['importance']:.4f}")

    # Check accuracy metrics
    assert "train_accuracy" in model_handler.results
    assert "test_accuracy" in model_handler.results
    assert model_handler.results["train_accuracy"] > 0.5
    assert model_handler.results["test_accuracy"] > 0.5

    # Check confusion matrix
    assert "confusion_matrix" in model_handler.results
    cm = model_handler.results["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2

    # Check feature importance
    assert "feature_importance" in model_handler.results
    feature_importance = model_handler.results["feature_importance"]
    assert len(feature_importance) == len(model_config["features"])

    # Verify feature importance values
    for feature in feature_importance:
        assert "feature" in feature
        assert "importance" in feature
        assert feature["importance"] >= 0
        assert feature["importance"] <= 1


@pytest.mark.model_training
def test_svm_classifier(student_data, model_config):
    """
    Test SVM classifier on student performance data.

    Verifies:
    1. Model can be trained successfully
    2. Training and test accuracy are reasonable (> 0.5)
    3. Confusion matrix is properly generated
    4. All metrics are within expected ranges
    """
    print("\n=== SVM Classifier Test ===")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        student_data["X"], student_data["y"], test_size=0.2, random_state=42
    )

    # Initialize model handler
    model_handler = ModelHandler(
        model_type="svm",
        features=model_config["features"],
        target=model_config["target"],
        normalize=model_config["normalize"],
    )

    # Train model
    success = model_handler.train(X_train, X_test, y_train, y_test)
    assert success, "Model training failed"

    # Verify model was trained
    assert model_handler.model is not None
    assert model_handler.results is not None

    # Print results
    print("\nModel Results:")
    print(f"Training Accuracy: {model_handler.results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {model_handler.results['test_accuracy']:.4f}")

    print("\nConfusion Matrix:")
    cm = model_handler.results["confusion_matrix"]
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")

    # Check accuracy metrics
    assert "train_accuracy" in model_handler.results
    assert "test_accuracy" in model_handler.results
    assert model_handler.results["train_accuracy"] > 0.5
    assert model_handler.results["test_accuracy"] > 0.5

    # Check confusion matrix
    assert "confusion_matrix" in model_handler.results
    cm = model_handler.results["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2


@pytest.mark.model_persistence
def test_model_saving(student_data, model_config):
    """
    Test model saving functionality with student performance data.

    Verifies:
    1. Model can be trained successfully
    2. Model can be saved to disk
    3. Saved model file exists
    4. Cleanup of test files works correctly
    """
    print("\n=== Model Saving Test ===")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        student_data["X"], student_data["y"], test_size=0.2, random_state=42
    )

    # Initialize model handler
    model_handler = ModelHandler(
        model_type="random_forest_classifier",
        features=model_config["features"],
        target=model_config["target"],
        normalize=model_config["normalize"],
    )

    # Train model
    model_handler.train(X_train, X_test, y_train, y_test)

    # Save model
    filepath = model_handler.save_model("student_performance_model")
    print(f"\nModel saved to: {filepath}")
    assert os.path.exists(filepath)

    # Save training data with timestamp
    data_name = get_timestamped_name("student_performance_data")
    data_path = os.path.join("data", f"{data_name}.csv")
    student_data["df"].to_csv(data_path, index=False)
    print(f"Training data saved to: {data_path}")

    # Clean up test files
    if os.path.exists(filepath):
        os.remove(filepath)
        print("Test model file cleaned up successfully")
    if os.path.exists(data_path):
        os.remove(data_path)
        print("Test data file cleaned up successfully")
