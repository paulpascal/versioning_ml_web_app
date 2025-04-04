import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from app.utils.model_handler import ModelHandler
import os


# Test data generation
def generate_test_data():
    """Generate synthetic test data for both regression and classification"""
    # Generate regression data
    n_samples = 100
    X_reg = np.random.randn(n_samples, 3)
    y_reg = (
        2 * X_reg[:, 0]
        + 0.5 * X_reg[:, 1]
        - 0.3 * X_reg[:, 2]
        + np.random.randn(n_samples) * 0.1
    )

    # Generate classification data
    X_clf = np.random.randn(n_samples, 3)
    y_clf = (X_clf[:, 0] + X_clf[:, 1] + X_clf[:, 2] > 0).astype(int)

    return X_reg, y_reg, X_clf, y_clf


@pytest.fixture
def test_data():
    """Fixture to provide test data"""
    X_reg, y_reg, X_clf, y_clf = generate_test_data()
    return {
        "regression": {"X": X_reg, "y": y_reg},
        "classification": {"X": X_clf, "y": y_clf},
    }


@pytest.fixture
def features():
    """Fixture to provide feature names"""
    return ["feature1", "feature2", "feature3"]


def test_data_loading():
    """Test if test data file exists and can be loaded"""
    test_data_path = "data/test_data.csv"
    assert os.path.exists(test_data_path), "Test data file not found"

    # Try to load the data
    try:
        df = pd.read_csv(test_data_path)
        assert not df.empty, "Test data file is empty"
        assert len(df.columns) > 0, "Test data has no columns"
    except Exception as e:
        pytest.fail(f"Failed to load test data: {str(e)}")


def test_linear_regression(test_data, features):
    """Test linear regression model training"""
    # Prepare data
    X = test_data["regression"]["X"]
    y = test_data["regression"]["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model handler
    model_handler = ModelHandler(
        model_type="linear_regression",
        features=features,
        target_column="target",
        normalize=True,
    )

    # Train model
    model_handler.train(X_train, X_test, y_train, y_test)

    # Check results
    assert model_handler.results is not None, "Training results are None"
    assert "mse" in model_handler.results, "MSE not found in results"
    assert "r2" in model_handler.results, "R² not found in results"
    assert model_handler.results["mse"] >= 0, "MSE should be non-negative"
    assert 0 <= model_handler.results["r2"] <= 1, "R² should be between 0 and 1"


def test_random_forest_classifier(test_data, features):
    """Test random forest classifier training"""
    # Prepare data
    X = test_data["classification"]["X"]
    y = test_data["classification"]["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model handler
    model_handler = ModelHandler(
        model_type="random_forest_classifier",
        features=features,
        target_column="target",
        normalize=True,
    )

    # Train model
    model_handler.train(X_train, X_test, y_train, y_test)

    # Check results
    assert model_handler.results is not None, "Training results are None"
    assert (
        "train_accuracy" in model_handler.results
    ), "Training accuracy not found in results"
    assert (
        "test_accuracy" in model_handler.results
    ), "Test accuracy not found in results"
    assert (
        "confusion_matrix" in model_handler.results
    ), "Confusion matrix not found in results"
    assert (
        0 <= model_handler.results["test_accuracy"] <= 1
    ), "Accuracy should be between 0 and 1"


def test_svm_classifier(test_data, features):
    """Test SVM classifier training"""
    # Prepare data
    X = test_data["classification"]["X"]
    y = test_data["classification"]["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model handler
    model_handler = ModelHandler(
        model_type="svm", features=features, target_column="target", normalize=True
    )

    # Train model
    model_handler.train(X_train, X_test, y_train, y_test)

    # Check results
    assert model_handler.results is not None, "Training results are None"
    assert (
        "train_accuracy" in model_handler.results
    ), "Training accuracy not found in results"
    assert (
        "test_accuracy" in model_handler.results
    ), "Test accuracy not found in results"
    assert (
        "confusion_matrix" in model_handler.results
    ), "Confusion matrix not found in results"
    assert (
        0 <= model_handler.results["test_accuracy"] <= 1
    ), "Accuracy should be between 0 and 1"


def test_feature_importance(test_data, features):
    """Test feature importance calculation"""
    # Prepare data
    X = test_data["classification"]["X"]
    y = test_data["classification"]["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model handler with Random Forest
    model_handler = ModelHandler(
        model_type="random_forest_classifier",
        features=features,
        target_column="target",
        normalize=True,
    )

    # Train model
    model_handler.train(X_train, X_test, y_train, y_test)

    # Check feature importance
    assert (
        "feature_importance" in model_handler.results
    ), "Feature importance not found in results"
    importance_list = model_handler.results["feature_importance"]
    assert len(importance_list) == len(
        features
    ), "Number of features in importance list doesn't match"
    assert all(
        imp["importance"] >= 0 for imp in importance_list
    ), "All importance scores should be non-negative"


def test_model_saving(test_data, features):
    """Test model saving functionality"""
    # Prepare data
    X = test_data["classification"]["X"]
    y = test_data["classification"]["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model handler
    model_handler = ModelHandler(
        model_type="random_forest_classifier",
        features=features,
        target_column="target",
        normalize=True,
    )

    # Train model
    model_handler.train(X_train, X_test, y_train, y_test)

    # Test model saving
    test_model_name = "test_model"
    try:
        model_handler.save_model(test_model_name)
        assert os.path.exists(
            f"models/{test_model_name}.joblib"
        ), "Model file not created"
    finally:
        # Clean up
        if os.path.exists(f"models/{test_model_name}.joblib"):
            os.remove(f"models/{test_model_name}.joblib")
